from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains.retrieval import create_retrieval_chain
from langchain import hub
from langchain_pinecone import PineconeVectorStore
from langchain_groq.chat_models import ChatGroq
from langchain.chains.combine_documents.stuff import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from typing import Any, Dict, List
load_dotenv()
import os

index_name = os.getenv("INDEX_NAME")


def run_llm(query: str, chat_history=None, model="llama3-8b-8192", temperature=0.0, max_tokens=1000):
    embiddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )
    vectorstore = PineconeVectorStore(
        embedding=embiddings,
        index_name=index_name,
    )
    
    # Use the parameters from the sidebar
    chat = ChatGroq(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    
    rephrase_prompt = hub.pull("langchain-ai/chat-langchain-rephrase")

    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    stuff_documents_chain = create_stuff_documents_chain(chat, retrieval_qa_chat_prompt)

    history_aware_retriever = create_history_aware_retriever(
        llm=chat, retriever=vectorstore.as_retriever(), prompt=rephrase_prompt
    )
    qa = create_retrieval_chain(
        retriever=history_aware_retriever, combine_docs_chain=stuff_documents_chain
    )

    result = qa.invoke(input={"input": query, "chat_history": chat_history})
    return result