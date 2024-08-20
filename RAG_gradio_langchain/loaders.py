from langchain_community.document_loaders import TextLoader, PyPDFLoader

loade = TextLoader("hihi.txt", encoding='utf-8')

def load_documents(file_path, mime_type):
    if mime_type == "application/pdf":
        loader = PyPDFLoader(file_path)
    else:
        loader = TextLoader(file_path, encoding='utf-8')
    return loader.load()
