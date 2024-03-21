from langchain.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from transformers import pipeline, BartTokenizerFast
from transformers import pipeline

import os,time
import streamlit as st

st.title("News Article Summarizer 📰")
st.write("Uncover Insights, Save Time ⏳")

st.sidebar.title("News Article URLs")


def get_summary(final_text):
   summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
   result = summarizer(final_text,
                    max_length = 1024, 
                    min_length = 100,
                    do_sample=False,
                    truncation=True)
   result = result[0]['summary_text']
   st.subheader(f"Summary for Article:")
   st.write(result)
#    return result


def text_preprocessing(u):
   loaders = UnstructuredURLLoader(u)
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
   return final_texts


# urls_1=[
#     "https://www.news.com.au/world/coronavirus/health/wear-a-mask-nsw-health-responds-to-a-rise-in-cases-in-light-of-new-subvariant-strains/news-story/90cad04f2a329d8730871c00b3dd00cc"
#     ]
url_input = st.text_area("URL(s)", height=100, help="Enter one or more news article URLs separated by line breaks.")
urls = []
for url in url_input.split("\n"):
        if url.strip() != "":
            urls.append([url])
print("url: ", urls)

# urls = []
# for i in range(2):
#    url = st.text_input(f"URL {0+i}")
#    urls.append([url])
#    print(urls[i])
if st.button("Process URL"):
    for i in range(len(urls)):
       r = text_preprocessing(urls[i])
       get_summary(r)
    #    st.subheader(f"Summary for Article {i+1}:")
    #    print("Summary 1: ",s)
    #    st.write(s)
       st.write(f"source: {urls[i]}")

# final_text_1 = text_preprocessing(urls_1)
#get_summary( final_text_1)
# print("Summary 1: ",get_summary(r))

# urls_2 =[
#     "https://www.news.com.au/finance/business/other-industries/liquidator-of-collapsed-building-firm-alleges-director-was-loaned-nearly-1-million-in-company-money/news-story/493daf5d91f6a6b1267d542d6fa5b184"
#     ]
# final_text_2 = text_preprocessing(urls_2)
# print("Summary 2: ",get_summary(final_text_2))


