from langchain.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import pipeline
import os,time
import streamlit as st

st.title("News Article Summarizer 📰")
st.write("Uncover Insights, Save Time ⏳")

st.sidebar.title("News Article URLs")

# Set up pipeline for summarization
summarizer = pipeline("summarization", model="abertsch/unlimiformer-bart-govreport-alternating") 
                    #   model="facebook/bart-large-cnn")

# Text input for user to enter URL(s)
# url_input = st.sidebar.text_area("URL(s)", height=100, help="Enter one or more news article URLs separated by line breaks.")
urls = []
# for i in range(2):
url = st.sidebar.text_input(f"URL {0}")
urls.append(url)


# main_placeholder = st.empty()
# main_placeholder.text("Data Loading...Started...✅✅✅")

def text_preprocessing(u):
  loaders = UnstructuredURLLoader(u)
  print("URL: ", type(u)," ", u)
  data = loaders.load()
  print(len(data))
  text_splitter = RecursiveCharacterTextSplitter(
      chunk_size=1000,
      chunk_overlap=200
      )
  # As data is of type documents we can directly use split_documents over split_text in order to get the chunks.
  docs = text_splitter.split_documents(data)
  print(len(docs))
  final_texts = ""
  for i in range(0,len(docs)):
    #print(i," : ", input_text[i].page_content)
    final_texts = final_texts + docs[i].page_content
    #print("final_texts : ", final_texts)
  return final_texts,u

def summarize_articles(text,urlss):
    print("url: ",urlss)
    print(text, "final_text_1 Generated...")
    summaries = []
    if urlss != "":
        try:
            main_placeholder.text("Generating...Summary...✅✅✅")
            article_summary = summarizer(text, max_length=1024, min_length=100, do_sample=False)[0]["summary_text"]
            summaries.append((urlss, article_summary))
        except Exception as e:
            st.error(f"Error summarizing article from {urlss}: {e}")
    return summaries
# final_text_1 = None
# ur = None
if st.sidebar.button("Process URLs"):
    main_placeholder = st.empty()
    main_placeholder.text("Data Loading...Started...✅✅✅")
    

if st.button("Summarize"):
    print("After button clicked...")
    try:
        final_text_1,ur = text_preprocessing(urls)
        # main_placeholder.text("Embedding Vector Started Building...✅✅✅")
        main_placeholder = st.empty()
        main_placeholder.text("Model Loading...Started...✅✅✅")
        summaries = summarize_articles(final_text_1,ur)
                            
        # Display article summaries
        for url, summary in summaries:
            st.subheader("Summary:")
            # st.subheader(f"Summary for article from {url}")
            st.write(summary)
            st.write(f"Source: {url}")
    except Exception as e:
        st.error(f"Error summarizing article: {e}")
                    