import gradio as gr
from chatbot import add_message, bot
from utils import clear_history


import gradio as gr
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.sessions import SessionMiddleware
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import bcrypt
import uuid
from starlette.requests import Request
from starlette.responses import RedirectResponse

clear_history()

from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import bcrypt

DATABASE_URL = "postgresql+psycopg2://postgres:12345@localhost/authorization"
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
app.add_middleware(SessionMiddleware, secret_key="your_secret_key")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return bcrypt.checkpw(plain_password.encode('utf-8'), hashed_password.encode('utf-8'))

def create_session():
    return str(uuid.uuid4())

def register_user(username: str, password: str):
    session = SessionLocal()
    user = session.query(User).filter(User.username == username).first()
    if user:
        session.close()
        return "Username already exists"
    
    password_hash = hash_password(password)
    new_user = User(username=username, password_hash=password_hash)
    session.add(new_user)
    session.commit()
    session.close()
    return "User registered successfully"

def check_user(username: str, password: str) -> bool:
    session = SessionLocal()
    user = session.query(User).filter(User.username == username).first()
    session.close()
    if user and verify_password(password, user.password_hash):
        return True
    return False

def login_user(username: str, password: str):
    if check_user(username, password):
        session_id = create_session()
        session = SessionLocal()
        user = session.query(User).filter(User.username == username).first()
        user.session_id = session_id
        session.commit()
        session.close()
        return f"Welcome, {username}!"
    else:
        return "Invalid username or password"

def handle_register(username, password):
    result = register_user(username, password)
    if result == "User registered successfully":
        result = login_user(username, password)
        if "Welcome" in result:
            return (gr.update(visible=False), gr.update(visible=False),
                    gr.update(visible=True), gr.update(visible=False),
                    gr.update(visible=False), result)
    return gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), result

def handle_login(username, password):
    result = login_user(username, password)
    if "Welcome" in result:
        return (gr.update(visible=False), gr.update(visible=False),
                gr.update(visible=True), gr.update(visible=False),
                gr.update(visible=False), result)
    return gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), result

def handle_logout():
    result = "Logged out successfully"
    return (gr.update(visible=True), gr.update(visible=True),
            gr.update(visible=False), result,
            gr.update(visible=True), gr.update(visible=True))

with gr.Blocks(fill_height=True) as demo:
    chatbot = gr.Chatbot(elem_id="chatbot", bubble_full_width=False, scale=1)
    chat_input = gr.MultimodalTextbox(interactive=True, file_count="multiple", placeholder="Enter message or upload file...", show_label=False)
    clear_button = gr.Button("🗑(Clear history)", size="sm")

    with gr.Row():
        with gr.Column():
            welcome_text = gr.Textbox(label="", visible=True, interactive=False)
            username_input = gr.Textbox(label="Username")
            password_input = gr.Textbox(label="Password", type="password")
            register_button = gr.Button("Register")
            login_button = gr.Button("Login")
            logout_button = gr.Button("Logout", visible=False)
    


    chat_msg = chat_input.submit(add_message, [chatbot, chat_input], [chatbot, chat_input])
    bot_msg = chat_msg.then(bot, chatbot, chatbot, api_name="bot_response")
    bot_msg.then(lambda: gr.MultimodalTextbox(interactive=True), None, [chat_input])

    clear_button.click(clear_history, [], [chatbot, chat_input])
    
    register_button.click(handle_register, [username_input, password_input],
                          [username_input, password_input, logout_button, register_button, login_button, welcome_text])
    login_button.click(handle_login, [username_input, password_input],
                       [username_input, password_input, logout_button, register_button, login_button, welcome_text])
    logout_button.click(handle_logout, [], 
                        [username_input, password_input, logout_button, welcome_text, register_button, login_button])

demo.launch(share=True)