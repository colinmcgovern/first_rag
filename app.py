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
        chunk_size=10000,
        chunk_overlap=1000
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


def generate_prompt(context, question):
    return f"""
    Use the following context to answer the question. Do not use anything else to answer the question.:
    {context}
    The question is: {question}
    """

def call_llm(docs,user_question):

    print("asdf")
    print(generate_prompt(docs[0].page_content, user_question))

    payload={
            "messages": [
                {
                    "role": "user",
                    "content": [{"text": generate_prompt(docs[0].page_content, user_question)}]
                }
            ]
        }
    body=json.dumps(payload)
    model_id="us.amazon.nova-micro-v1:0"
    response=bedrock.invoke_model_with_response_stream(
        body=body,
        modelId=model_id,
        # accept="application/json",
        # contentType="application/json"
    )
    # response_body=json.loads(next(response.get("body")).read())
    # response_text=response_body['generation']
    # print(response_text)
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


# def get_response_llm(llm,vectorstore_faiss,query):
#     qa = RetrievalQA.from_chain_type(
#         llm=llm,
#         chain_type="stuff",
#         retriever=vectorstore_faiss.as_retriever(
#             search_type="similarity",
#             search_kwargs={"k": 3}
#         ),
#         return_source_documents=True,
#     )
#     result = qa({"query": query})
#     chunks = [doc.page_content for doc in result["source_documents"]]
#     return chunks


# prompt = PromptTemplate(
#     input_variables=["context", "question"],
#     template="""You are an assistant that answers questions based on the given context.

# Context:
# {context}

# Question:
# {question}

# Answer:"""
# )

def get_relevant_answer(llm, vectorstore_faiss, query, k=3):
    # Step 1: Retrieve relevant chunks
    retriever = vectorstore_faiss.as_retriever(search_kwargs={"k": k})
    docs = retriever.get_relevant_documents(query)
    return docs

def main():
    st.set_page_config("Chat PDF")
    st.header("Chat with PDF")

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
            faiss_index = FAISS.load_local("faiss_index", bedrock_embeddings, allow_dangerous_deserialization=True)
            llm=get_micro_llm()
            st.write(get_relevant_answer(llm,faiss_index,user_question)[0].page_content)
            #st.write(get_response_llm(llm,faiss_index,user_question)[0].page_content)
            st.write("-----")
            st.write(
                call_llm(
                    get_relevant_answer(llm,faiss_index,user_question),
                    user_question
                )
            )
            st.success("Answer Generated Successfully")

if(__name__=="__main__"):
    main()