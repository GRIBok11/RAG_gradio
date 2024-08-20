import plotly.express as px

def random_plot():
    df = px.data.iris()
    fig = px.scatter(df, x="sepal_width", y="sepal_length", color="species",
                    size='petal_length', hover_data=['petal_width'])
    return fig

def update_request_count(request_count, max_requests):
    return f"{request_count}/{max_requests}"
