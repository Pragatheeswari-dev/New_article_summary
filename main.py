import os
import streamlit as st
import pickle
import time
import langchain
import langchain_community
from langchain import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
#from langchain.embeddings import OpenAIEmbeddings
from langchain_community.embeddings import OpenAIEmbeddings
#from langchain.vectorstores import FAISS
from langchain_community.vectorstores import FAISS
import openai
#from langchain_community.llms import OpenAI
#from langchain_openai import OpenAI
import transformers
#from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import pipeline

#summarizer = pipeline("summarization", model="facebook/bart-large-cnn")


from dotenv import load_dotenv
load_dotenv()  # take environment variables from .env (especially openai api key)

st.title("News Research Tool 📈")
st.sidebar.title("News Article URLs")



urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")
#file_path = "faiss_store_openai.pkl"

main_placeholder = st.empty()
llm = OpenAI(model_name='gpt-3.5-turbo',temperature=0, max_tokens=500)
#tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
#model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")
#summary_clicked = st.button("Get Summary")

if process_url_clicked:
    # load data
    loader = UnstructuredURLLoader(urls=urls)
    main_placeholder.text("Data Loading...Started...✅✅✅")
    data = loader.load()
    # split data
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=1000,
        chunk_overlap=200
    )
    main_placeholder.text("Text Splitter...Started...✅✅✅")
    docs = text_splitter.split_documents(data)
    # create embeddings and save it to FAISS index
    embeddings = OpenAIEmbeddings()
    vectorstore_openai = FAISS.from_documents(docs, embeddings)
    main_placeholder.text("Embedding Vector Started Building...✅✅✅")
    time.sleep(2)

    vectorstore_openai.save_local("faiss_store")

    # Save the FAISS index to a pickle file
    #with open(file_path, "wb") as f:
    #    pickle.dump(vectorstore_openai, f)

query = main_placeholder.text_input("Question: ")
if query:
    vectorstore_openai = FAISS.load_local("faiss_store", OpenAIEmbeddings(),allow_dangerous_deserialization = True)
    chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever = vectorstore_openai.as_retriever())
    #chain = RetrievalQAWithSourcesChain.from_llm(llm = llm, retriever = vectorstore_openai.as_retriever())
    result = chain({"question": query}, return_only_outputs=True)
    # result will be a dictionary of this format --> {"answer": "", "sources": [] }
    st.header("Answer")
    st.write(result["answer"])

    # Display sources, if available
    sources = result.get("sources", "")
    if sources:
        st.subheader("Sources:")
        sources_list = sources.split("\n")  # Split the sources by newline
        for source in sources_list:
            st.write(source)


    

