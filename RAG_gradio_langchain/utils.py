import plotly.express as px
from retriever import retriever
import gradio as gr

def random_plot():
    df = px.data.iris()
    fig = px.scatter(df, x="sepal_width", y="sepal_length", color="species",
                    size='petal_length', hover_data=['petal_width'])
    return fig

def update_request_count(request_count, max_requests):
    return f"{request_count}/{max_requests}"

def clear_history():
    from retriever import vectorstore
    try:
        for collection in vectorstore._client.list_collections():
            ids = collection.get()['ids']
            print('REMOVE %s document(s) from %s collection' % (str(len(ids)), collection.name))
            if len(ids): collection.delete(ids)
        from loaders import loade
        do = loade.load()
        retriever.add_documents(do) 
    except Exception as e:
        print(f"Error clearing history: {e}")
    return [], gr.MultimodalTextbox(value=None, interactive=True)