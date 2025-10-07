import json
import os
import sys
import boto3

from langchain_aws import BedrockEmbeddings
from langchain_community.llms.bedrock import Bedrock
from langchain_community.chat_models import BedrockChat

import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader

from langchain_community.vectorstores import FAISS

from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

import streamlit as st

bedrock=boto3.client(service_name="bedrock-runtime")
bedrock_embeddings=BedrockEmbeddings(
    model_id="amazon.titan-embed-text-v1",
    client=bedrock
)

def data_ingestion():
    loader=PyPDFDirectoryLoader("pdfs/")
    documents=loader.load()

    text_splitter=RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    docs=text_splitter.split_documents(documents)
    return docs

def get_vector_store(docs):

    vector_store=FAISS.from_documents(docs, bedrock_embeddings)
    vector_store.save_local("faiss_index")

def get_micro_llm():
    llm=Bedrock(
        model_id="us.amazon.nova-micro-v1:0",
        client=bedrock,
        model_kwargs={"maxTokens":512,"stopSequences":[],"temperature":0.7,"topP":0.9}
    )
    return llm


def prompt_lynch(context_docs, question):

    prompt = "The user has the following investment idea:\n"
    prompt += question + "\n"
    prompt += "Use the following context from peter lynch's investment books "
    prompt += "to say why the users investment idea is good and bad.\n"
    prompt += "\n"

    doc_iter = 0

    for doc in context_docs:
        doc_iter = doc_iter + 1
        prompt += "context " + str(doc_iter) + ":\n"
        prompt += doc.page_content + "\n"

    return prompt

def call_llm(context_docs,user_question):

    payload={
            "messages": [
                {
                    "role": "user",
                    "content": [{"text": prompt_lynch(context_docs, user_question)}]
                }
            ]
        }
    body=json.dumps(payload)
    model_id="us.amazon.nova-micro-v1:0"
    response=bedrock.invoke_model_with_response_stream(
        body=body,
        modelId=model_id,
    )
    response_text = ""
    for event in response['body']:
        chunk = event.get("chunk")
        if chunk:
            chunks = chunk["bytes"].decode("utf-8")
            try:
                parsed_chunk = json.loads(chunks)
                if "contentBlockDelta" in parsed_chunk and "delta" in parsed_chunk["contentBlockDelta"] and "text" in parsed_chunk["contentBlockDelta"]["delta"]:
                    chunks = parsed_chunk["contentBlockDelta"]["delta"]["text"]
                    response_text += chunks
            except json.JSONDecodeError:
                pass  # Keep original chunks if it's not valid JSON
    return response_text

def get_relative_docs(llm, vectorstore_faiss, query, k=5):
    # Step 1: Retrieve relevant chunks
    retriever = vectorstore_faiss.as_retriever(search_kwargs={"k": k})
    docs = retriever.get_relevant_documents(query)
    return docs

def main():
    st.set_page_config("Peter Lynch Investment Strategy", layout="wide")
    st.header("Peter Lynch Investment Strategy")

    user_question = st.text_input("Ask your question here:")

    with st.sidebar:
        st.title("Update or create vector store:")

        if st.button("Vectors Update"):
            with st.spinner("Ingesting Data..."):
                docs=data_ingestion()
                get_vector_store(docs)
                st.success("Vector Store Updated Successfully")
        
    if st.button("Get Answer"):
        with st.spinner("Getting Answer..."):
            faiss_index = FAISS.load_local(
                "faiss_index",
                bedrock_embeddings,
                allow_dangerous_deserialization=True
            )
            llm=get_micro_llm()
            # Show relevant context in an expander
            relevant_docs = get_relative_docs(llm, faiss_index, user_question)
            for doc in relevant_docs:
                with st.expander("Relevant Context", expanded=False):
                    st.write(f"Filename: {doc.metadata.get('source', 'Unknown')}")
                    st.write(f"Page: {doc.metadata.get('page', 'Unknown')}")
                    st.write("Content:")
                    st.write(doc.page_content)
            
            with st.expander("Prompt", expanded=False):
                st.write(prompt_lynch(relevant_docs, user_question))

            st.write(
                call_llm(
                    relevant_docs,
                    user_question
                )
            )
            st.success("Answer Generated Successfully")

if(__name__=="__main__"):
    main()