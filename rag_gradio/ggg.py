import os
import shutil
import atexit
from langchain_groq import ChatGroq
from langchain.schema import AIMessage, HumanMessage
import gradio as gr
import plotly.express as px
from dotenv import load_dotenv
from langchain.chains.summarize import load_summarize_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.storage import InMemoryStore
from langchain.retrievers import ParentDocumentRetriever
from operator import itemgetter

load_dotenv()
groq_api_key = os.getenv('groq_api_key')

llm = ChatGroq(
    temperature=0.7,
    groq_api_key=groq_api_key,
    model_name="mixtral-8x7b-32768"
)

# Укажите директорию для сохранения файлов
UPLOAD_DIRECTORY = "uploaded_files"

# Создайте директорию, если она не существует
if not os.path.exists(UPLOAD_DIRECTORY):
    os.makedirs(UPLOAD_DIRECTORY)

def random_plot():
    df = px.data.iris()
    fig = px.scatter(df, x="sepal_width", y="sepal_length", color="species",
                    size='petal_length', hover_data=['petal_width'])
    return fig

def add_message(history, message):
    if history is None:
        history = []
    if message["files"]:
        for file in message["files"]:
            # Сохранение файла в указанную директорию
            local_path = os.path.join(UPLOAD_DIRECTORY, os.path.basename(file))
            shutil.copy(file, local_path)
            history.append((f"File uploaded: {local_path}", None))
            # Добавление файла в retriever
            loader = TextLoader(local_path, encoding='utf-8')
            docs = loader.load()
            retriever.add_documents(docs, ids=None)
    if message["text"] is not None:
        history.append((message["text"], None))
    return history, gr.MultimodalTextbox(value=None, interactive=False)

def bot(history):
    if history is None:
        history = []
    history_langchain_format = []
    for human, ai in history:
        if human is not None:
            history_langchain_format.append(HumanMessage(content=human))
        if ai is not None:
            history_langchain_format.append(AIMessage(content=ai))
    gpt_response = llm.invoke(history_langchain_format)
    history[-1][1] = gpt_response.content
    return history

def cleanup_files():
    if os.path.exists(UPLOAD_DIRECTORY):
        shutil.rmtree(UPLOAD_DIRECTORY)

# Регистрация функции очистки при завершении программы
atexit.register(cleanup_files)

def read_all_files(directory):
    file_contents = []
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path):
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                file_contents.append(content)
    return file_contents

def create_chain(retriever):
    chain_sum = load_summarize_chain(llm, chain_type="refine")

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful Q&A helper for the documentation, trained to answer questions from the Master of Orion manual."
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

embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
docs = []
if os.path.exists(UPLOAD_DIRECTORY):
    # Загрузка всех файлов из директории
    for file_path in os.listdir(UPLOAD_DIRECTORY):
        full_path = os.path.join(UPLOAD_DIRECTORY, file_path)
        if os.path.isfile(full_path):
            loader = TextLoader(full_path, encoding='utf-8')
            docs.extend(loader.load())

# Создание и настройка retriever
child_splitter = RecursiveCharacterTextSplitter(chunk_size=300)
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=1500)
vectorstore = Chroma(collection_name="split_parents", embedding_function=embedding_function)
store = InMemoryStore()

retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=store,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
)

if docs:
    retriever.add_documents(docs, ids=None)

def process_question(question):
    chain = create_chain(retriever)
    result = chain({"question": question})
    return result["answer"]

fig = random_plot()

with gr.Blocks(fill_height=True) as demo:
    chatbot = gr.Chatbot(
        elem_id="chatbot",
        bubble_full_width=False,
        scale=1,
    )

    chat_input = gr.MultimodalTextbox(interactive=True,
                                      file_count="multiple",
                                      placeholder="Enter message or upload file...", show_label=False)

    chat_msg = chat_input.submit(add_message, [chatbot, chat_input], [chatbot, chat_input])
    bot_msg = chat_msg.then(bot, chatbot, chatbot, api_name="bot_response")
    bot_msg.then(lambda: gr.MultimodalTextbox(interactive=True), None, [chat_input])

    question_input = gr.Textbox(label="Question", placeholder="Enter your question here...")
    question_input.submit(process_question, question_input, chatbot)

demo.launch()
