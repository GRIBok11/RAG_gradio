import psycopg2
from langchain_huggingface import HuggingFaceEmbeddings

# Параметры подключения
conn = psycopg2.connect(
    host="localhost",    # Локальный хост для связи с Docker
    port="4444",         # Порт PostgreSQL
    database="rag_gradio_db",  # Имя базы данных
    user="postgres"         # Пароль (если установлен)
)

embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

from langchain_core.documents import Document
from langchain_postgres import PGVector
from langchain_postgres.vectorstores import PGVector
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
from typing import List, Tuple

# See docker command above to launch a postgres instance with pgvector enabled.
connection = "postgresql+psycopg://postgres:@localhost:4444/rag_gradio_db"  # Uses psycopg3!
collection_name = "my_docs"

engine = create_engine(connection)
SessionLocal = sessionmaker(bind=engine)


def delete_collection(name) -> None:
        vectorstore = PGVector(
            embeddings=embedding_function,
            collection_name=name,
            connection=connection,
            use_jsonb=True,
            pre_delete_collection= True,
        )
      


def clear_history(username: str):
    try:
        session = SessionLocal()
        #collection_name = f"vectorstores/{username}"
        # Примерный SQL-запрос для удаления всех векторов из коллекции
        session.execute(f"""
                DELETE FROM langchain_pg_embedding
                WHERE collection_id IN (
                    SELECT uuid FROM langchain_pg_collection WHERE name = {collection_name}
                );
            """)
        session.commit()
            
        print(f"Данные, связанные с коллекцией {collection_name}, удалены.")

       
        session.close()
    except Exception as e:
        print(f"Ошибка при очистке истории: {e}")
    


delete_collection("vectorstores/1")