import gradio as gr
import plotly.express as px
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_groq import ChatGroq
from langchain.schema import AIMessage, HumanMessage
from dotenv import load_dotenv
import os
import shutil
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import Chroma
from langchain.chains.summarize import load_summarize_chain
from langchain.storage import InMemoryStore
from langchain.retrievers import ParentDocumentRetriever
from langchain_core.prompts import ChatPromptTemplate
from operator import itemgetter
import mimetypes
import sqlite3
# Chatbot demo with multimodal input (text, markdown, LaTeX, code blocks, image, audio, & video). Plus shows support for streaming text.

load_dotenv()
groq_api_key = os.getenv('groq_api_key')

llm = ChatGroq(
    temperature=0.7,
    groq_api_key=groq_api_key,
    model_name="mixtral-8x7b-32768"
)
embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# This text splitter is used to create the child documents
child_splitter = RecursiveCharacterTextSplitter(chunk_size=400)
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000)

# The vectorstore to use to index the child chunks
vectorstore = Chroma(collection_name="split_parents", embedding_function=embedding_function,persist_directory="vectore")
store = InMemoryStore()

retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=store,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
)

loade = TextLoader("hihi.txt", encoding='utf-8')
do = loade.load()
retriever.add_documents(do)

eval_llm = ChatGroq(
    temperature=0,
    groq_api_key=groq_api_key,
    model_name="Llama3-70b-8192"
)

request_count = 0
max_requests = 20
history = []
def random_plot():
    df = px.data.iris()
    fig = px.scatter(df, x="sepal_width", y="sepal_length", color="species",
                    size='petal_length', hover_data=['petal_width'])
    return fig

def create_chain(retriever):
    chain_sum = load_summarize_chain(eval_llm, chain_type="refine")

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Use the following pieces of context to answer the question at the end. Generate the answer based on the given context only if you find the answer in the context. If you do not find any information related to the question in the given context, just say that you don't know, don't try to make up an answer. Keep your answer expressive."
                "\n\nThe relevant documents will be retrieved in the following messages.",
            ),
            ("system", "{context}"),
            ("human", "{question}"),
        ]
    )

    response_generator = prompt | llm | StrOutputParser()
    chain = (
        {
            "context": itemgetter("question") | retriever | chain_sum,
            "question": itemgetter("question"),
        }
        | response_generator
    )
    return chain

chain = create_chain(retriever)

def add_message(history, message):
    global request_count, doc_ids
    if request_count >= max_requests:
        return history, gr.MultimodalTextbox(value="Request limit reached. Please clear history.", interactive=False)
    
    if history is None:
        history = []
    
    if not message["text"] and not message["files"]:
        return history, gr.MultimodalTextbox(value="Please enter a message or upload a file.", interactive=True)
    
    for x in message["files"]:
        mime_type, encoding = mimetypes.guess_type(x)

        if mime_type == "application/pdf":
            loader = PyPDFLoader(x)
            docs = loader.load()
        else:
            loader = TextLoader(x, encoding='utf-8')
            docs = loader.load()
        
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

    for collection in vectorstore._client.list_collections():
        ids = collection.get()['ids']
        print('REMOVE %s document(s) from %s collection' % (str(len(ids)), collection.name))
        if len(ids): collection.delete(ids)
    loade = TextLoader("hihi.txt", encoding='utf-8')
    do = loade.load()
    retriever.add_documents(do) 
    return [], gr.MultimodalTextbox(value=None, interactive=True)

fig = random_plot()




# Пример использования функции

with gr.Blocks(fill_height=True) as demo:
    chatbot = gr.Chatbot(
        elem_id="chatbot",
        bubble_full_width=False,
        scale=1,
    )

    chat_input = gr.MultimodalTextbox(interactive=True,
                                      file_count="multiple",
                                      placeholder="Enter message or upload file...", show_label=False)
    clear_button = gr.Button("🗑", size="sm")

    chat_msg = chat_input.submit(add_message, [chatbot, chat_input], [chatbot, chat_input])
    bot_msg = chat_msg.then(bot, chatbot, chatbot, api_name="bot_response")
    bot_msg.then(lambda: gr.MultimodalTextbox(interactive=True), None, [chat_input])
    clear_button.click(clear_history, [], [chatbot, chat_input])

    gr.Markdown(f"{request_count}/{max_requests}")

demo.launch(share=True)
