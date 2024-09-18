import gradio as gr
from langchain_community.document_loaders import TextLoader
import bcrypt
import json
from typing import List, Tuple, Any
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

def save_history(history: List[Tuple[str, str]], username: str) -> None:
    with open(f"vectorstores/{username}/{username}_history.json", "w") as file:
        json.dump(history, file)

def load_history(username: str) -> List[Tuple[str, str]]:
    try:
        with open(f"vectorstores/{username}/{username}_history.json", "r") as file:
            history = json.load(file)
        return history
    except FileNotFoundError:
        return []

from sqlalchemy.orm import sessionmaker, Session
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

def create_new_chat(username: str, history: List[Tuple[str, str]]) -> None:
    session = SessionLocal()
    user = session.query(User).filter(User.username == username).first()
    
    if user:
        chat_history = ChatHistory(user_id=user.id, chat_content=json.dumps(history))
        session.add(chat_history)
        session.commit()
    id1 = chat_history.id
    session.close()
    return history, id1



def save_chat_history(username: str, chat_id: int, history: List[Tuple[str, str]]) -> None:
    session = SessionLocal()
    user = session.query(User).filter(User.username == username).first()
    
    if user:
        # Ищем запись чата по chat_id и проверяем, что этот чат принадлежит пользователю
        chat_history = session.query(ChatHistory).filter(ChatHistory.id == chat_id, ChatHistory.user_id == user.id).first()
        
        if chat_history:
            # Обновляем содержимое чата
            chat_history.chat_content = json.dumps(history)
            session.commit()
        else:
            print(f"No chat found with ID {chat_id} for user {username}")
    
    session.close()

def load_last_chat(username: str) -> Tuple[List[Tuple[str, str]], int]:
    session = SessionLocal()
    user = session.query(User).filter(User.username == username).first()
    
    if user:
        # Получаем последний созданный чат
        last_chat = session.query(ChatHistory).filter(ChatHistory.user_id == user.id).order_by(ChatHistory.created_at.desc()).first()
        if last_chat:
            history = json.loads(last_chat.chat_content)
            session.close()
            return history, last_chat.id  # Возвращаем историю и chat_id последнего чата
    session.close()
    return create_new_chat(username,[["non","non"]])




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
    chat = session.query(ChatHistory).filter(ChatHistory.id == chat_id).first()
    
    if chat:
        session.close()
        return json.loads(chat.chat_content)
    session.close()
    return []


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return bcrypt.checkpw(plain_password.encode('utf-8'), hashed_password.encode('utf-8'))


def load_previous_history(username):
        history = load_history(username)
        return history, gr.MultimodalTextbox(value=None, interactive=True)






def clear_history(vectorstore,retriever) -> Tuple[List[Tuple[str, str]], gr.MultimodalTextbox]:
    try:
        for collection in vectorstore._client.list_collections():
            ids = collection.get()['ids']
            print('REMOVE %s document(s) from %s collection' % (str(len(ids)), collection.name))
            if len(ids): collection.delete(ids)
        loade = TextLoader("hihi.txt", encoding='utf-8')
        do = loade.load()
        retriever.add_documents(do)
    except Exception as e:
        print(f"Error clearing history: {e}")
    return [], gr.MultimodalTextbox(value=None, interactive=True)

