
from langchain.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import pipeline
import os,time
import streamlit as st

# st.title("News Article Summarizer 📰")
st.sidebar.title("Uncover Insights, Save Time")
# st.caption('This is a string that explains something above.')
st.sidebar.caption('Provide the _news article_ :blue[URL links] below and emojis :sunglasses:')


# Set up pipeline for summarization
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Define UI layout
st.title("News Article Summarizer 📰")
st.write("Uncover Insights, Save Time ⏳")
st.divider()

# Text input for user to enter URL(s)
url_input = st.text_area("URL(s)", height=100, help="Enter one or more news article URLs separated by line breaks.")

# Function to summarize articles
def summarize_articles(urls):
    summaries = []
    for url in urls.split("\n"):
        if url.strip() != "":
            try:
                # Perform summarization
                article_summary = summarizer(url, max_length=1024, min_length=100, do_sample=False)[0]["summary_text"]
                summaries.append((url, article_summary))
            except Exception as e:
                st.error(f"Error summarizing article from {url}: {e}")
    return summaries

# Button to trigger summarization
if st.button("Summarize"):
    # Split URLs by line breaks and summarize each article
    summaries = summarize_articles(url_input)
    
    # Display article summaries
    for url, summary in summaries:
        st.subheader(f"Summary for article from {url}")
        st.write(summary)
