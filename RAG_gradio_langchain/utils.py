import gradio as gr
from langchain_community.document_loaders import TextLoader
import bcrypt
import json
from typing import List, Tuple, Any

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

