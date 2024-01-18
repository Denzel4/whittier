__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import pandas as pd
import numpy as np
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PDFMinerPDFasHTMLLoader
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import RetrievalQA
from langchain.indexes import VectorstoreIndexCreator
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import PromptLayerChatOpenAI
from langchain_community.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage
import gradio as gr
import openai
import os
from langchain_openai import OpenAIEmbeddings
import json

with open('secrets.json') as f:
    secrets = json.load(f)
#
api_key = secrets['api']
chat = PromptLayerChatOpenAI(openai_api_key=api_key)
chat = ChatOpenAI(openai_api_key=api_key,temperature=0.1, model_name="gpt-3.5-turbo")

from rag_data import TextLoader

file_path = 'data'
#data/contact.txt
#file_path ='https://raw.githubusercontent.com/Denzel4/whittier/main/data/contact.txt?token=GHSAT0AAAAAACMVVH47UKX7WBF2DJ2VEZ3UZNJAKVA'
#/home/cicero/wshc/whittier/data
text_file_paths = [os.path.join(file_path, path) for path in os.listdir(file_path)]

text_loader = TextLoader(text_file_paths)
loaded_data = text_loader.load()

if loaded_data:
    # Process the loaded data as needed
    print("Data successfully loaded.")

    # Chunk the loaded data (assuming chunk size is 100 characters)
    #chunk_size = 1000
    #for document in loaded_data:

     #   chunks = TextLoader.chunk_data(document, chunk_size)
      #  for chunk in chunks:
            # Process each chunk as needed
       #     print("Processing chunk:", chunk)

else:
    print("No data loaded.")

# Model embeddings initialization
embeddings = OpenAIEmbeddings(openai_api_key=api_key)

#text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
#chunk the  data
chunks = [TextLoader.chunk_data(document, 1000) for document in loaded_data]
# Create vector db
whit_db = Chroma.from_texts(loaded_data,embeddings)
#create an access to the db to access the  vector ,index
whit_retriever = whit_db.as_retriever(search_type="similarity",search_kwargs={"k":2})
#test Query
query = " How do I contact a member of my care team?"
def get_best(query):
  return whit_retriever.get_relevant_documents(query)

# Response function
def get_response(query):
  qa = RetrievalQA.from_chain_type(
    llm=OpenAI(openai_api_key=api_key), chain_type="map_reduce", retriever=whit_retriever, return_source_documents=True)
  #response = qa({'query':query})
  docs_score = whit_db.similarity_search_with_score(query=query, k = 3)
  response = qa({'query':query})
  whit_output = response['result']
  return whit_output#response['result']

def testtbot(input_text):
    response = chat([HumanMessage(content="As a WSHC Assistant, my main goal is to provide exceptional customer service and \
    assist with any inquiries or concerns regarding the facility. I have been extensively trained by WSHC engineers and have\
     access to detailed information about the facility. With a dedication to customer satisfaction, commitment to quality\
      care, and emphasis on community partnership, I am here to help. Please feel free to ask any questions you may have,\
       and I will do my best to assist you in a helpful and professional manner.\
    use {query} as context in answering the questions.Be helpful and professional.".format(query=get_response(input_text)))])
    return response.content
def chatbot(input_text):
    context = get_best(input_text)
    #print(f"Input text: {input_text}")s
    #print(f"Context: {context}")

    response = chat([HumanMessage(content="{query}".format(query=context))])
    print(f"Chat response: {response.content}")

    return response.content

iface = gr.Interface(fn=chatbot,
                     inputs=gr.components.Textbox(lines=7, label="Enter your text"),
                     outputs="text",
                     title="WSHC Assistant")
iface.launch(share=False,server_name="0.0.0.0", server_port=7860)