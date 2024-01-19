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
chat = ChatOpenAI(openai_api_key=api_key,temperature=0.0, model_name="gpt-4-turbo")

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
whit_retriever = whit_db.as_retriever(search_type="similarity",search_kwargs={"k":1})

def get_best(query):
  return whit_retriever.get_relevant_documents(query)


def chatbot(input_text):
    context = get_best(input_text)
    response = chat([HumanMessage(content="As a WSHC assistant. Your role is to answer to enquiries about the facility and its services.\
                                  You are were trained by WSHC engineers.\
                                  Use this information {query} in answering".format(query=context))])
  

    return response.content

iface = gr.Interface(fn=chatbot,
                     inputs=gr.components.Textbox(lines=7, label="Enter your text"),
                     outputs="text",
                     title="WSHC Assistant")
iface.launch(share=False,server_name="0.0.0.0", server_port=7860)