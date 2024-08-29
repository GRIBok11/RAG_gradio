from langchain_community.document_loaders import TextLoader, PyPDFLoader
from typing import List, Union

loade: TextLoader = TextLoader("hihi.txt", encoding='utf-8')

def load_documents(file_path: str, mime_type: str) -> List[Union[str, dict]]:
    try:
        if mime_type == "application/pdf":
            loader = PyPDFLoader(file_path)
        else:
            loader = TextLoader(file_path, encoding='utf-8')
        return loader.load()
    except Exception as e:
        print(f"Error loading document: {e}")
        return []
