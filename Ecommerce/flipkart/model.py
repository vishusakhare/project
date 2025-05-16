import pandas as pd
from langchain_core.documents import Document

def dataconverter():

    product_data = pd.read_csv(r"flipkart_data.csv")

    data = product_data[["product_title", "review"]]

    product_list = []

    ## Itrate over the rows of the DataFrame

    for index, row in data.iterrows():
        object = {
            "product_name": row["product_title"],
            "review": row["review"]
        }

    ## Append the object to the product list
    product_list.append(object)
    docs = []
    for entry in product_list:
        metadata = {"product_name": entry['product_name']}
        doc = Document(page_content= entry['review'], metadata= metadata)
        docs.append(doc)    
    return docs

#data_ingestion.py
from langchain_astradb import AstraDBVectorStore
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
import os
from flipkart.data_converter import dataconverter
from dotenv import load_dotenv
load_dotenv()

GROQ_API_KEY=os.getenv("GROQ_API_KEY")
ASTRA_DB_API_ENDPOINT=os.getenv("ASTRA_DB_API_ENDPOINT")
ASTRA_DB_APPLICATION_TOKEN=os.getenv("ASTRA_DB_APPLICATION_TOKEN")
ASTRA_DB_KEYSPACE=os.getenv("ASTRA_DB_KEYSPACE")
HF_TOKEN = os.getenv("HF_TOKEN")


GROQ_API_KEY=os.getenv("GROQ_API_KEY")
ASTRA_DB_API_ENDPOINT=os.getenv("ASTRA_DB_API_ENDPOINT")
ASTRA_DB_APPLICATION_TOKEN=os.getenv("ASTRA_DB_APPLICATION_TOKEN")
ASTRA_DB_KEYSPACE=os.getenv("ASTRA_DB_KEYSPACE")
HF_TOKEN = os.getenv("HF_TOKEN")

embedding = HuggingFaceInferenceAPIEmbeddings(api_key= HF_TOKEN, model_name="BAAI/bge-base-en-v1.5")

def data_ingestion(status):

    vstore = AstraDBVectorStore(
        embedding=embedding,
        collection_name = "flipkart",
        api_endpoint = ASTRA_DB_API_ENDPOINT,
        token = ASTRA_DB_APPLICATION_TOKEN,
        namespace = ASTRA_DB_KEYSPACE 
    )

    storage = status

    if storage == None:
        docs = dataconverter()
        insert_ids = vstore.add_documents(docs)
    
    else:
        return vstore
    return vstore, insert_ids

if __name__ == "__main__":

    vstore, insert_ids = data_ingestion(None)
    print(f"\n Inserted {len(insert_ids)} documents.")
    results = vstore.similarity_search("Can you tell me the low budget sound basshead?")
    for res in results:
        print(f"\n {res.page_content} [{res.metadata}]")

#retrieval_generation.py
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import MessagesPlaceholder
from langchain.chains import create_history_aware_retriever
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from flipkart.data_ingestion import data_ingestion



from dotenv import load_dotenv
import os

load_dotenv()

os.environ["GROQ_API_KEY"]= os.getenv("GROQ_API_KEY")

model = ChatGroq(model="llama-3.1-70b-versatile", temperature=0.5)




chat_history= []
store = {}
def get_session_history(session_id: str)-> BaseChatMessageHistory:
  if session_id not in store:
    store[session_id]= ChatMessageHistory()
  return store[session_id]


def generation(vstore):
    retriever = vstore.as_retriever(search_kwargs={"k": 3})

    retriever_prompt = ("Given a chat history and the latest user question which might reference context in the chat history,"
    "formulate a standalone question which can be understood without the chat history."
    "Do NOT answer the question, just reformulate it if needed and otherwise return it as is."
    )

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
    ("system", retriever_prompt),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    ]
)
    history_aware_retriever = create_history_aware_retriever(model, retriever, contextualize_q_prompt)
    PRODUCT_BOT_TEMPLATE = """
    Your ecommercebot bot is an expert in product recommendations and customer queries.
    It analyzes product titles and reviews to provide accurate and helpful responses.
    Ensure your answers are relevant to the product context and refrain from straying off-topic.
    Your responses should be concise and informative.

    CONTEXT:
    {context}

    QUESTION: {input}

    YOUR ANSWER:

    """
    qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", PRODUCT_BOT_TEMPLATE),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ]
)
    question_answer_chain = create_stuff_documents_chain(model, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)
    return conversational_rag_chain



if __name__ == "__main__":
   vstore = data_ingestion("done")
   conversational_rag_chain = generation(vstore)
   answer= conversational_rag_chain.invoke(
    {"input": "can you tell me the best bluetooth buds?"},
    config={
        "configurable": {"session_id": "dhruv"}
    },  # constructs a key "abc123" in store.
)["answer"]
   print(answer)
   answer1= conversational_rag_chain.invoke(
    {"input": "what is my previous question?"},
    config={
        "configurable": {"session_id": "dhruv"}
    },  # constructs a key "abc123" in store.
)["answer"]
   print(answer1)

#app.py
from flask import Flask, render_template, request
from flipkart.retrieval_generation import generation
from flipkart.data_ingestion import data_ingestion


vstore = data_ingestion("done")
chain = generation(vstore)


app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")



@app.route("/get", methods = ["POST", "GET"])
def chat():
   
   if request.method == "POST":
      msg = request.form["msg"]
      input = msg

      result = chain.invoke(
         {"input": input},
    config={
        "configurable": {"session_id": "dhruv"}
    },
)["answer"]

      return str(result)

if __name__ == '__main__':
    
    app.run(host="0.0.0.0", port=5000, debug= True)

#setup.py
from setuptools import find_packages, setup
from typing import List


def get_requirements() ->List[str]:

    """
    This function will return list of requirements
    """
    requirement_list:List[str] = []

    """
    Write a code to read requirements.txt file and append each requirements in requirement_list variable.
    """
    return requirement_list

setup(
    name = "flipkart",
    version= "0.0.1",
    author="Dhruv-Saxena",
    author_email="dhruvsaxena.uk@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements()
)

#template.py
import os 
from pathlib import Path

project_name = "flipkart"

list_of_files = [

    f"{project_name}/__init__.py",
    f"{project_name}/data_converter.py",
    f"{project_name}/data_ingestion.py",
    f"{project_name}/retrieval_generation.py",
    "static/style.css",
    "templates/chat.html",
    "setup.py",
    "app.py",
    "requirements.txt",
    ".env"
]


for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)

    if filedir !="":
        os.makedirs(filedir, exist_ok=True)
    
    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) ==0):
        with open(filepath, "w") as f:
            pass