import gradio as gr
from langchain.schema import AIMessage, HumanMessage
from loaders import load_documents
from retriever import retriever
from chain import chain
import mimetypes

request_count = 0
max_requests = 20
history = []

def add_message(history, message):
    global request_count
    if request_count >= max_requests:
        return history, gr.MultimodalTextbox(value="Request limit reached. Please clear history.", interactive=False)
    
    if history is None:
        history = []
    
    if not message["text"] and not message["files"]:
        return history, gr.MultimodalTextbox(value="Please enter a message or upload a file.", interactive=True)
    
    for x in message["files"]:
        mime_type, encoding = mimetypes.guess_type(x)
        docs = load_documents(x, mime_type)
        retriever.add_documents(docs, ids=None)
        history.append((x, None))

    if message["text"] is not None:
        history.append((message["text"], None))
    
    request_count += 1
    return history, gr.MultimodalTextbox(value=None, interactive=False)

def bot(history):
    global request_count
    if request_count >= max_requests:
        return history
    
    if history is None:
        history = []
    history_langchain_format = []
    for human, ai in history:
        if human is not None:
            history_langchain_format.append(HumanMessage(content=human))
        if ai is not None:
            history_langchain_format.append(AIMessage(content=ai))
    gpt_response = chain.invoke({"question": history[-1][0]})
    history[-1][1] = gpt_response  # Обрабатываем строковый ответ
    
    request_count += 1
    return history

def clear_history():
    from retriever import vectorstore
    for collection in vectorstore._client.list_collections():
        ids = collection.get()['ids']
        print('REMOVE %s document(s) from %s collection' % (str(len(ids)), collection.name))
        if len(ids): collection.delete(ids)
    from loaders import loade
    do = loade.load()
    retriever.add_documents(do) 
    return [], gr.MultimodalTextbox(value=None, interactive=True)
