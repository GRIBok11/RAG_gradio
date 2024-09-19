import gradio as gr
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_groq import ChatGroq
from langchain.schema import AIMessage, HumanMessage
from dotenv import load_dotenv
import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain.chains.summarize import load_summarize_chain
from langchain.storage import InMemoryStore
from langchain.retrievers import ParentDocumentRetriever
from langchain_core.prompts import ChatPromptTemplate
from operator import itemgetter
import mimetypes
from sqlalchemy.ext.declarative import declarative_base
from typing import List, Tuple, Any, Optional
from transformers import GPT2Tokenizer
from utils import save_history , save_chat_history

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
MAX_TOKENS = 7000  
load_dotenv()
groq_api_key = os.getenv('groq_api_key')

llm = ChatGroq(
    temperature=0.7,
    groq_api_key=groq_api_key,
    model_name="mixtral-8x7b-32768"
)
embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

child_splitter = RecursiveCharacterTextSplitter(chunk_size=400)
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000)

store = InMemoryStore()

eval_llm = ChatGroq(
    temperature=0,
    groq_api_key=groq_api_key,
    model_name="Llama3-70b-8192"
)

history = []
def count_tokens(history: List[Tuple[str, str]]) -> int:
    total_tokens = 0
    for human, ai in history:
        if human is not None:
            total_tokens += len(tokenizer.encode(human))
        if ai is not None:
            total_tokens += len(tokenizer.encode(ai))
    return total_tokens


def bot(history: List[Tuple[str, Optional[str]]], 
        retriever: ParentDocumentRetriever, 
        name: str) -> List[Tuple[str, Optional[str]]]:
    max_tokens: int = 5000
    if history is None:
        history = []
    history_langchain_format = []
    for human, ai in history:
        if human is not None:
            history_langchain_format.append(HumanMessage(content=human))
        if ai is not None:
            history_langchain_format.append(AIMessage(content=ai))
    

    chain = create_chain(retriever)
    try:
        gpt_response = chain.invoke({"question": history[-1][0]})
        history[-1][1] = gpt_response
    except Exception as e:
        # Логируем ошибку и возвращаем сообщение о переполнении контекста
        print(f"Error during chain invocation: {e}")
        history[-1][1] = f"Error during chain invocation: {e}"
        return history

    
    history[-1][1] = gpt_response

    
    print("tokens")
    print(count_tokens(history))
    if count_tokens(history) > max_tokens:
        history.pop()
        return history, gr.MultimodalTextbox(value="Token limit exceeded. Please clear history.", interactive=False)
    save_history(history, name)
    save_chat_history(name, history)
    return history

def add_message(history: List[Tuple[str, Optional[str]]], 
                message: dict, 
                retriever_state: ParentDocumentRetriever, 
                name: str) -> Tuple[List[Tuple[str, Optional[str]]], gr.components.MultimodalTextbox]:

    
    if history == []:
        print("hdfnom[dhfgnjmolfhgjnob;]")
        loade = TextLoader("hihi.txt", encoding='utf-8')
        do = loade.load()
        retriever_state.add_documents(do,ids=None)
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
        
        retriever_state.add_documents(docs, ids=None)
        history.append((x, None))


    if message["text"] is not None:
        history.append((message["text"], None))
  
    save_history(history, name)
    save_chat_history(name, history)
    return history, gr.MultimodalTextbox(value=None, interactive=False)

def create_chain(retriever: ParentDocumentRetriever) -> Tuple[Any, int]:
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

