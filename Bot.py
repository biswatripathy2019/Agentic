#import webscrap
#import lib
import os
import requests
import json
from typing import List
#from bs4 import BeautifulSoup
from IPython.display import Markdown, display, update_display
from openai import OpenAI
import langchain_community
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import CharacterTextSplitter
from openai import OpenAI
#from langchain.schema import Document
from langchain_core.documents.base import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
import glob
import gradio as gr
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from dotenv import load_dotenv
load_dotenv()
MODEL = "gpt-4o-mini"
db_name = "vector_db"
import os
import sys
openai=OpenAI(api_key=os.getenv('openai_api_key'))

# imports
# If these fail, please check you're running from an 'activated' environment with (llms) in the command prompt


# Read in documents using LangChain's loaders
# Take everything in all the sub-folders of our knowledgebase
drive='/home/biswatripathy21/gopath/pkg/mod/router_on_chat'
#sys.argv[1]
db_name='vector'
#sys.argv[2]
folders = glob.glob(drive+"*")
print(drive+"*")

# With thanks to CG and Jon R, students on the course, for this fix needed for some users
#text_loader_kwargs = {'encoding': 'utf-8'}
# If that doesn't work, some Windows users might need to uncomment the next line instead
text_loader_kwargs={'autodetect_encoding': True}
print(folders)
documents = []
for folder in folders:
    # Check if the item is a directory before processing it
    print('print folder',folder)
    if os.path.isdir(folder):
      doc_type = os.path.basename(folder)
      loader = DirectoryLoader(folder, glob="**/*.md", loader_cls=TextLoader, 
      loader_kwargs=text_loader_kwargs)
      folder_docs = loader.load()
      print('folder_docs', folder_docs)
      for doc in folder_docs:
        doc.metadata["doc_type"] = doc_type
        documents.append(doc)
print('documents',documents)    
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(documents)
doc_types = set(chunk.metadata['doc_type'] for chunk in chunks)
print(f"Document types found: {', '.join(doc_types)}")
# Put the chunks of data into a Vector Store that associates a Vector Embedding with each chunk
embeddings = OpenAIEmbeddings(api_key=os.getenv('openai_api_key'))
if os.path.exists(db_name):
  Chroma(persist_directory=db_name, embedding_function=embeddings).delete_collection()
# Create our Chroma vectorstore!

vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=db_name)
print(f"Vectorstore created with {vectorstore._collection.count()} documents")
collection = vectorstore._collection
sample_embedding = collection.get(limit=1, include=["embeddings"])["embeddings"][0]
dimensions = len(sample_embedding)
print(f"The vectors have {dimensions:,} dimensions")
# create a new Chat with OpenAI
llm = ChatOpenAI(temperature=0.7, model_name=MODEL,api_key=os.getenv('openai_api_key'))

# set up the conversation memory for the chat
memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)

# the retriever is an abstraction over the VectorStore that will be used during RAG
retriever = vectorstore.as_retriever()

# putting it together: set up the conversation chain with the GPT 4o-mini LLM, the vector store and memory
conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, memory=memory)
def chat(message,history):
    result = conversation_chain.invoke({"question": message})
    return result["answer"]
def main():
  # More involved Gradio code as we're not using the preset Chat interface!
# Passing in inbrowser=True in the last line will cause a Gradio window to pop up immediately.
  system_message='you are an AI assistance chatbot.always reply with voice message'
  gr.ChatInterface(chat).launch(inbrowser=True,debug=True,share=True)
if __name__ == "__main__":
  main()
  #pip install "numpy<2.0" 
