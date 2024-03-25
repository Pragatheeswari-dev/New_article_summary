from langchain.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
# from transformers import pipeline, BartTokenizerFast
from transformers import pipeline

import os,time
import streamlit as st

st.title("News Article Summarizer 📰")
st.write("  **Uncover Insights, Save Time ⏳**")
# st.caption("Use the sidebar to learn more about the tool and get started!")


# st.sidebar.title("News Article URLs")
st.sidebar.title("About the News Article Summarizer:")
st.sidebar.write(
        """
        The demo tool designed to help you quickly extract key insights and main points from news articles. 
        It is implemented using LangChain framework.
         - load the text and split into chunks of desired size. 
         - 
         By simply providing the URL(s) of the articles you want to summarize, the tool uses advanced natural language processing (NLP) techniques to generate concise summaries, saving you time and effort.

        ### Features:
        - *Efficient Summarization*: Instantly summarize news articles with just a click of a button.
        - *Accurate Insights*: Extract main points and key information from lengthy articles with high accuracy.
        - *Multi-URL Support*: Summarize multiple articles at once by entering their URLs separated by line breaks.
        - *Easy to Use*: User-friendly interface makes it simple for anyone to utilize the tool without any technical expertise.

        ### How It Works:
        1. *Enter URL(s)*: Provide the URL(s) of the news article(s) you want to summarize.
        2. *Generate Summary*: Click the "Summarize" button to initiate the summarization process.
        3. *Review Results*: Read the concise summaries generated for each article and gain quick insights.

        ### Try It Now:
        Paste the URL(s) of news articles you want to summarize in the input box and click "Summarize" to uncover the main points and save time reading through lengthy articles.
        """
    )


def get_summary(final_text):
   
   main_placeholder.text("Embedding Vector Started Building...✅✅✅")
   summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
   
   main_placeholder.text("Generating...Summary...✅✅✅")
   result = summarizer(final_text,
                    max_length = 1024, 
                    min_length = 100,
                    do_sample=False,
                    truncation=True)
   result = result[0]['summary_text']
   st.subheader(f"Summary:")
#    st.text_area(result, height=120)
   st.write(result)


def text_preprocessing(u):
   main_placeholder.text("Data Loading...Started...✅✅✅")
   loaders = UnstructuredURLLoader(u)
   data = loaders.load()
   print(len(data))
   text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
      chunk_size=1000,
      chunk_overlap=200
      )
   docs = text_splitter.split_documents(data)
   print(len(docs))
   final_texts = ""
   for i in range(0,len(docs)):
    
    final_texts = final_texts + docs[i].page_content
    
   return final_texts


default_urls = [
    "https://www.governmentnews.com.au/queensland-says-no-to-new-olympic-stadium/",
    "https://www.pm.gov.au/media/parents-and-economy-benefit-latest-reform",
    "https://www.news.com.au/world/coronavirus/health/wear-a-mask-nsw-health-responds-to-a-rise-in-cases-in-light-of-new-subvariant-strains/news-story/90cad04f2a329d8730871c00b3dd00cc"
]

# url_input = st.text_area("URL(s)", height=200, help="Enter one or more news article URLs separated by line breaks.")
st.subheader("Try It Out:")
st.write("Paste the URL(s) of news articles you want to summarize below, or use the sample URLs provided:")

url_input = st.text_area("URL(s)", value="\n".join(default_urls), height=200, help="Enter one or more news article URLs separated by line breaks. You can also use the sample URLs provided below.")
urls = []
for url in url_input.split("\n"):
        if url.strip() != "":
            urls.append([url])
print("url: ", urls)


if st.button("Process URL"):
    main_placeholder = st.empty()
    # main_placeholder.text("Data Loading...Started...✅✅✅")
    for i in range(len(urls)):
       r = text_preprocessing(urls[i])
    #    main_placeholder.text("Embedding Vector Started Building...✅✅✅")
       main_placeholder.text("Model Loading...Started...✅✅✅")
       get_summary(r)
       st.write(f"source: {urls[i]}")