import gradio as gr
from langchain.schema import AIMessage, HumanMessage
from loaders import load_documents
from retriever import retriever
from chain import chain
import mimetypes
from transformers import GPT2Tokenizer
import json
from typing import List, Tuple, Any

def save_history(history: List[Tuple[str, str]], username: str) -> None:
    with open(f"{username}_history.json", "w") as file:
        json.dump(history, file)

def load_history(username: str) -> List[Tuple[str, str]]:
    try:
        with open(f"{username}_history.json", "r") as file:
            history = json.load(file)
        return history
    except FileNotFoundError:
        return []

request_count: int = 0
max_requests: int = 20
history: List[Tuple[str, str]] = []

def count_tokens(history: List[Tuple[str, str]]) -> int:
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    total_tokens = 0
    for human, ai in history:
        if human is not None:
            total_tokens += len(tokenizer.encode(human))
        if ai is not None:
            total_tokens += len(tokenizer.encode(ai))
    return total_tokens

def add_message(history: List[Tuple[str, str]], message: dict, username: str) -> Tuple[List[Tuple[str, str]], gr.MultimodalTextbox]:
    global request_count
  
    if request_count >= max_requests:
        return history, gr.MultimodalTextbox(value="Request limit reached. Please clear history.", interactive=False)
    
    if history is None:
        history = []
    
    if not message["text"] and not message["files"]:
        return history, gr.MultimodalTextbox(value="Please enter a message or upload a file.", interactive=True)
    
    for x in message["files"]:
        try:
            mime_type, encoding = mimetypes.guess_type(x)
            docs = load_documents(x, mime_type)
            retriever.add_documents(docs, ids=None)
            history.append((x, None))
        except Exception as e:
            print(f"Error processing file {x}: {e}")
            continue

    if message["text"] is not None:
        history.append((message["text"], None))
    
    save_history(history, username)
    return history, gr.MultimodalTextbox(value=None, interactive=False)

def bot(history: List[Tuple[str, str]], username: str) -> List[Tuple[str, str]]:
    max_tokens: int = 5000
    global request_count
    request_count = len(history)
    print(len(history))
    print("tokens")
    print(count_tokens(history))
    if count_tokens(history) > max_tokens:
        history.pop()
        return history, gr.MultimodalTextbox(value="Token limit exceeded. Please clear history.", interactive=False)
    if request_count >= max_requests:
        return history
    
    if history is None:
        history = []
    history_langchain_format: List[Any] = []
    for human, ai in history:
        if human is not None:
            history_langchain_format.append(HumanMessage(content=human))
        if ai is not None:
            history_langchain_format.append(AIMessage(content=ai))
    try:
        gpt_response: str = chain.invoke({"question": history[-1][0]})
        gpt_response += f"\n\nRequest Count: {request_count}/{max_requests}"
        request_count += 0.5
        history[-1][1] = gpt_response  # Обрабатываем строковый ответ
    except Exception as e:
        print(f"Error generating response: {e}")
        history[-1][1] = f"An error occurred while generating the response.\n\nRequest Count: {request_count}/{max_requests}"
    save_history(history, username)
    return history

def clear_history() -> Tuple[List[Tuple[str, str]], gr.MultimodalTextbox]:
    from retriever import vectorstore
    try:
        for collection in vectorstore._client.list_collections():
            ids = collection.get()['ids']
            print('REMOVE %s document(s) from %s collection' % (str(len(ids)), collection.name))
            if len(ids): collection.delete(ids)
        from loaders import loade
        do = loade.load()
        retriever.add_documents(do) 
    except Exception as e:
        print(f"Error clearing history: {e}")
    return [], gr.MultimodalTextbox(value=None, interactive=True)
