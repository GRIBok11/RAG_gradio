import gradio as gr
import plotly.express as px
from langchain_community.document_loaders import TextLoader
from dotenv import load_dotenv
import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.storage import InMemoryStore
from langchain.retrievers import ParentDocumentRetriever
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.sessions import SessionMiddleware
import bcrypt
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session


from chatbot import add_message,bot
from utils import clear_history,verify_password,load_previous_history




def authenticate_user(username: str, password: str) -> bool:
    session: Session = SessionLocal()
    user: User = session.query(User).filter(User.username == username).first()
    session.close()

    if user and verify_password(password, user.password_hash):
        return True
    return False


load_dotenv()
DATABASE_URL: str = os.getenv('DATABASE_URL')
engine = create_engine(DATABASE_URL)
Base = declarative_base()

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True, nullable=False)
    password_hash = Column(String, nullable=False)
    session_id = Column(String, unique=True, nullable=True)

Base.metadata.create_all(bind=engine)

app = FastAPI()

secret_key = os.urandom(32)
app.add_middleware(SessionMiddleware, secret_key=secret_key)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)




def random_plot():
    df = px.data.iris()
    fig = px.scatter(df, x="sepal_width", y="sepal_length", color="species",
                    size='petal_length', hover_data=['petal_width'])
    return fig
















# Настройки интерфейса Gradio
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
    cl_button = gr.Button("qwer", size="sm")

    
    user_name_state = gr.State("")
    vectorstore_state = gr.State(None)
    retriever_state = gr.State(None)

    def update_message(request: gr.Request):
        username = request.username  # Получаем имя пользователя
        user_directory = f"vectorstores/{username}"  # Директория для vectorstore
        embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

        child_splitter = RecursiveCharacterTextSplitter(chunk_size=400)
        parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000)

        store = InMemoryStore()



        history = []
        # Создаем директорию пользователя, если её нет
        os.makedirs(user_directory, exist_ok=True)
        
        # Создаем vectorstore для пользователя
        vectorstore = Chroma(
            collection_name="split_parents",
            embedding_function=embedding_function,
            persist_directory=user_directory
        )
        
        # Создаем retriever для пользователя
        retriever = ParentDocumentRetriever(
            vectorstore=vectorstore,
            docstore=store,
            child_splitter=child_splitter,
            parent_splitter=parent_splitter,
        )
        
        # Загружаем примерные данные для retriever (если необходимо)
        loade = TextLoader("hihi.txt", encoding='utf-8')
        docs = loade.load()
        retriever.add_documents(docs)
        
        # Загружаем историю чата для пользователя
        history, _ = load_previous_history(username)
        
        return f"Welcome, {username}", history, username, vectorstore, retriever
    
    m = gr.Textbox(label="Username")

    demo.load(update_message, [], [m, chatbot, user_name_state, vectorstore_state, retriever_state])

    def af(username):
        print(username)

    chat_msg = chat_input.submit(add_message, [chatbot, chat_input, retriever_state,user_name_state], [chatbot, chat_input])
    bot_msg = chat_msg.then(bot, [chatbot,  retriever_state,user_name_state], chatbot, api_name="bot_response")
    bot_msg.then(lambda: gr.MultimodalTextbox(interactive=True), None, [chat_input])
    clear_button.click(clear_history, [vectorstore_state,retriever_state], [chatbot, chat_input])
    logout_button = gr.Button("Logout", link="/logout")
    cl_button.click(af, user_name_state)



demo.launch(server_port=4444, auth=authenticate_user , share= True)
