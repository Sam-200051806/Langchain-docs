from langchain_huggingface import HuggingFaceEmbeddings
# from langchain.document_loaders import ReadTheDocsLoader
from langchain_community.document_loaders import ReadTheDocsLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
load_dotenv()
import os
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)


def ingest_docs():
    doc_path = "langchain-docs/api.python.langchain.com/en/latest"
    loader = ReadTheDocsLoader(doc_path, encoding="utf-8")
    raw_documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    documents = text_splitter.split_documents(raw_documents)
    
    # Process documents in batches of 100
    batch_size = 100
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i + batch_size]
        print(f"Processing batch {i//batch_size + 1}/{(len(documents)-1)//batch_size + 1}")
        
        PineconeVectorStore.from_documents(
            embedding=embeddings,
            documents=batch,
            index_name="langchain-doc-index",
        )
    
    print("**** Ingestion complete ****")





if __name__ == "__main__":
    ingest_docs()
    print("Ingestion complete.")
