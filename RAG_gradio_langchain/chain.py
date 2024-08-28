from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter
from config import groq_api_key
from retriever import retriever  # Import retriever

sum_llm = ChatGroq(
    temperature=0,
    groq_api_key=groq_api_key,
    model_name="Llama3-70b-8192"
)

llm = ChatGroq(
        temperature=0.7,
        groq_api_key=groq_api_key,
        model_name="mixtral-8x7b-32768"
    )




def create_chain(retriever):
    chain_sum = load_summarize_chain(sum_llm, chain_type="refine")

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

chain = create_chain(retriever)
