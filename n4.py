from langchain.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from transformers import BartTokenizer, TFBartForConditionalGeneration, BartConfig
# from transformers import pipeline, BartTokenizerFast
from transformers import pipeline

import os,time
import streamlit as st
from langchain.chains import create_tagging_chain, create_tagging_chain_pydantic
from langchain_openai import ChatOpenAI
from langchain_core.pydantic_v1 import BaseModel, Field

# def pop_default(s):
#     s.pop('default')
import datetime
from datetime import datetime
import dateutil.parser as dparser

from dotenv import load_dotenv
# load_dotenv()  # take environment variables from .env (especially openai api key)
os.environ['OPENAI_API_KEY'] = st.secrets['OPENAI_API_KEY']

import requests
from bs4 import BeautifulSoup

import datetime
from datetime import datetime
import dateutil.parser as dparser

st.title("News Article Summarizer 📰")
st.write("  **Uncover Insights, Save Time ⏳**")
# st.caption("Use the sidebar to learn more about the tool and get started!")


# st.sidebar.title("News Article URLs")
st.sidebar.title("About:")
st.sidebar.write(
        """
        This demo tool is designed to uncover the main points and save time reading through lengthy articles. 
         - It is implemented using *LangChain framework*. 
         - Data from the url is loaded and splitted into desired chunks using *RecursiveCharacterTextSplitter*.
         - To generate extractive summary, pre-trained transformer model *'bart-large-cnn'* is used.
         - Articles were *tagged (for sentiments, aggressiveness, language, style)* using openai, *'gpt-3.5-turbo-0613'*.
         - Title, Author Name, Date where retrived from the url using *'BeautifulSoup'* package commomly used for web scrapping.
         
        ### Features:
        - *Efficient Summarization*
        - *Accurate Insights*
        - *Multi-URL Support*
        - *Easy to Use*

        ### Further optimisation:
        - with use of vector databases (ability to store and retrieve large word embeddings), LLM can be optimised for several usecases.
            - **Community Well-being**: Summarizing public health articles to provide timely updates and recommendations for community health and safety.
            - **Policy Development**: Analyzing and summarizing policy documents to identify key issues and trends, facilitating evidence-based decision-making.
            - **Governance**: Summarizing news articles and reports related to government activities and public affairs to monitor public sentiment and inform policy responses.

         
        
        """
    )

@st.cache_resource  # 👈 Add the caching decorator
def load_model():
   model = TFBartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
   tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
   return pipeline("summarization", model=model, tokenizer=tokenizer)

summarizer = load_model()

def get_summary(final_text,summarizer):
   
   # main_placeholder.text("Embedding Vector Started Building...✅✅✅")
   # summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
   
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
   text_splitter = RecursiveCharacterTextSplitter(
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
    "https://www.news.com.au/entertainment/celebrity-life/celebrity-deaths/ive-sadly-diedbestselling-author-announces-own-death/news-story/f7b145a9070d832b43946a126cfc3e4e",
   #  "https://www.pm.gov.au/media/parents-and-economy-benefit-latest-reform",
   #  "https://www.news.com.au/world/coronavirus/health/wear-a-mask-nsw-health-responds-to-a-rise-in-cases-in-light-of-new-subvariant-strains/news-story/90cad04f2a329d8730871c00b3dd00cc"
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

llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613")

title = ""
author = ""
date = ""
summary = ""
class Tags(BaseModel):
   sentiment: str = Field(..., enum=["happy", "neutral", "sad"])
   aggressiveness: int = Field(
      ...,
      description="describes how aggressive the statement is, the higher the number the more aggressive",
      enum=[1, 2, 3, 4, 5],
   )
   language: str = Field(
      ..., enum=["spanish", "english", "french", "german", "italian"]
   )
   # political_tendency: str
   style: str = Field(..., enum = ["formal","informal"])
   # style: str = Field(..., enum = ["formal","informal"])
   # title: str = Field(title)
   # author: str = Field(author) 
   # date: str = Field(str(date))
   # summary: str  = Field(summary)

   # def  __init__(self):
   #    self.sentiment = "neutral"
   #    self.aggressiveness = 3
   #    self.language = "english"
   #    self.summary = summary
   #    self.title = title
   #    self.author = author
   #    self.date = date
      



if st.button("Process URL"):
   main_placeholder = st.empty()
   # main_placeholder.text("Data Loading...Started...✅✅✅")
   for i in range(len(urls)):
      print("url: ", urls[i][0])
      soup = BeautifulSoup(requests.get(urls[i][0]).content, 'html.parser')
      t = soup.find('title')
      title = t.get_text()
      author1 = soup.find('meta', {'name': 'author'})
      print("author1 = ", author1)
      if author1 is not None:
         author = author1["content"]
      else:
         author = " "
      print("author = ", author)
      datest = soup.find('meta', {'property': 'article:published_time'})
      if datest is not None:
         da = datest["content"]
         date = dparser.parse(da,fuzzy=True)
         date = str(date.date())
      else:
         date = " " 
      print("da = ", date)
      r = text_preprocessing(urls[i])
      #main_placeholder.text("Embedding Vector Started Building...✅✅✅")
      main_placeholder.text("Model Loading...Started...✅✅✅")
      summary = get_summary(r,summarizer)
      st.write(f"Title: {title}")
      st.write(f"Author: {author}")
      st.write(f"Date: {date}")
      # st.write(f"Title: {title}")
      chain = create_tagging_chain_pydantic(Tags, llm)
      # Tags.author = author
      # Tags.title = title
      # Tags.date = date
      # Tags.summary = summary
      # print(chain.run(summary))
      # st.write(chain.run("Estoy muy enojado con vos! Te voy a dar tu merecido!"))
      # print("r =  /n", r)
      st.write(chain.run(r))
      st.write(f"source: {urls[i]}")
      