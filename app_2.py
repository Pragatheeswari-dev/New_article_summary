
from langchain.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import pipeline
import os,time
import streamlit as st

st.title("News Article Summary Tool 📈")
st.sidebar.title("News Article URLs")

# i=0
urls = []
# url1 = st.sidebar.text_input(f"URL {i+1}")
# # print("URL: ", type(url)," ", url)
# urls.append(url1)
# url1 = st.sidebar.text_input(f"URL {i+1}")
# url1 = st.sidebar.text_input(f"URL {i+1}")

urls = []
for i in range(2):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append([url])
print("URL: ", type(urls)," ", urls)
process_url_clicked = st.sidebar.button("Process URLs")


def get_summary(final_text):
  # pipe_sum = pipeline(
  #       'summarization',
  #       model = model,
  #       tokenizer = tokenizer
  #       )
  #main_placeholder = st.empty()
  #main_placeholder.text("Model Loading...Started...✅✅✅")
  summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
  
  main_placeholder.text("Generating...Summary...✅✅✅")
  time.sleep(2)
  result = summarizer(final_text,
                        max_length = 1024, 
                        min_length = 100,
                        do_sample=False,
                        truncation=True)
  result1 = result[0]['summary_text']

#   int_val = st.slider('Summary Size', min_value=25, max_value=100, value=50, step=25)
#   print(int_val, " : ", type(int_val))
#   m = st.selectbox("Summary Size",(25,50,75,100), index=None, placeholder = "Select the Size of summary to be generated")
#   int_val = st.number_input('Summary Size', min_value=25, max_value=100, value=50, step=25)
  
#   m = main_placeholder.text_input("Enter Summary length: ")
#   m = int(min_len)
#   getsummary_clicked = st.button("Get Summary")

#   if getsummary_clicked: 
#      if m is None:
#         m = 100
#      else:
#         m = int(m)
#   if  m:
#      print("m = ",m)
     
#      main_placeholder.text("Generating...Summary...✅✅✅")
#      result = summarizer(final_text,
#                         max_length = 1024, 
#                         min_length = m,
#                         do_sample=False,
#                         truncation=True)
#      result1 = result[0]['summary_text']
#      return result1
     
  return result1
     

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

if process_url_clicked:
    if url.startswith("http"):
        main_placeholder = st.empty()
        main_placeholder.text("Data Loading...Started...✅✅✅")
        final_text_1 = text_preprocessing(urls)
        main_placeholder.text("Embedding Vector Started Building...✅✅✅")
        # time.sleep(2)

        # getsummary_clicked = st.button("Get Summary")
        # print("final_text_1 before cllick: ", final_text_1)

        # if getsummary_clicked:
        # print("final_text_1: ", final_text_1)
        main_placeholder.text("Model Loading...Started...✅✅✅")
        summary = get_summary(final_text_1)
        time.sleep(2)
        st.header("Summary")
        print("Summary 1: ",summary)
        st.write(summary)
