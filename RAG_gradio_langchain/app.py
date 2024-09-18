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
from utils import clear_history,verify_password,load_previous_history,load_last_chat, create_new_chat
import json



from typing import List, Tuple, Any

def authenticate_user(username: str, password: str) -> bool:
    session: Session = SessionLocal()
    user: User = session.query(User).filter(User.username == username).first()
    session.close()

    if user and verify_password(password, user.password_hash):
        return True
    return False

def save_chat_history(username: str, history) -> None:
    session = SessionLocal()
    user = session.query(User).filter(User.username == username).first()
    

    chat_history = ChatHistory(user_id=user.id, chat_content="hi")
    session.add(chat_history)
    session.commit()
    session.close()







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



from sqlalchemy import DateTime
from datetime import datetime

class ChatHistory(Base):
    __tablename__ = "chat_history"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, nullable=False)
    chat_content = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.now)  # Добавляем дату и время создания



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



def on_load_chat_click(chat_id: int, username: str):
    # Загрузить выбранный чат
    history = load_selected_chat(chat_id)
    #vectorstore = load_existing_vectorstore(username, chat_id)
    return history
    
def random_plot():
    df = px.data.iris()
    fig = px.scatter(df, x="sepal_width", y="sepal_length", color="species",
                    size='petal_length', hover_data=['petal_width'])
    return fig

def get_user_chats(username: str):
    session = SessionLocal()
    user = session.query(User).filter(User.username == username).first()
    
    if user:
        chats = session.query(ChatHistory).filter(ChatHistory.user_id == user.id).all()
        chat_list = [{"id": chat.id, "created_at": chat.created_at.strftime("%Y-%m-%d %H:%M:%S")} for chat in chats]
        session.close()
        return chat_list
    session.close()
    return []

def load_selected_chat(chat_id: int) -> List[Tuple[str, str]]:
    session = SessionLocal()
    chat = session.query(ChatHistory).filter(ChatHistory.user_id == chat_id).first()
    
    if chat:
        session.close()
        return json.loads(chat.chat_content)
    session.close()
    return [["non","non"]]

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
    #chat_list_dropdown = gr.Dropdown(label="Select a chat to load", choices=[])
    new_chat_button = gr.Button("New Chat")
    
    chat_id_state = gr.State(None)  # Добавляем state для хранения chat_id
    user_name_state = gr.State("")
    vectorstore_state = gr.State(None)
    retriever_state = gr.State(None)
    asd = gr.State([["non","non"]])

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
        #chat_list = get_user_chats(username)
        #chat_choices = [f"Chat {chat['id']} - {chat['created_at']}" for chat in chat_list]

        # Загружаем примерные данные для retriever (если необходимо)
        loade = TextLoader("hihi.txt", encoding='utf-8')
        docs = loade.load()
        retriever.add_documents(docs)
        
        history, chat_id = load_last_chat(username)
        return f"Welcome, {username}", history, username, chat_id, vectorstore, retriever#, chat_choices
    m = gr.Textbox(label="Username")

    demo.load(update_message, [], [m, chatbot, user_name_state, vectorstore_state, retriever_state])

    def af(username):
        save_chat_history(username, "hi")
        print(username)

    

    chat_msg = chat_input.submit(add_message, [chatbot, chat_input, retriever_state,user_name_state, chat_id_state], [chatbot, chat_input])
    bot_msg = chat_msg.then(bot, [chatbot, retriever_state, user_name_state, chat_id_state], chatbot, api_name="bot_response")
    bot_msg.then(lambda: gr.MultimodalTextbox(interactive=True), None, [chat_input])
    #chat_list_dropdown.change(on_load_chat_click, inputs=[chat_list_dropdown, user_name_state], outputs=[chatbot])
    clear_button.click(clear_history, [vectorstore_state, retriever_state], [chatbot, chat_input])
    new_chat_button.click(create_new_chat, inputs=[user_name_state,asd], outputs=[chatbot, chat_id_state])
    logout_button = gr.Button("Logout", link="/logout")
    cl_button.click(af, user_name_state)



demo.launch(server_port=4444, auth=authenticate_user , share=True)
