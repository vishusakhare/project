# 1. Scrape Data
from bs4 import BeautifulSoup
import requests

url = "https://www.flipkart.com/search?q=smartphones"
r = requests.get(url)
soup = BeautifulSoup(r.text, "lxml")

products = soup.find_all("div", class_="DOjaWF gdgoEp")
product_data = [p.get_text() for p in products]

# 2. Embed
from langchain.embeddings import HuggingFaceEmbeddings

embed = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 3. Store in Vector DB
from langchain_astradb import AstraDBVectorStore

vectordb = AstraDBVectorStore(
    embedding=embed,
    collection_name="flipkart",
    api_endpoint=ASTRA_DB_API_ENDPOINT,
    token=ASTRA_DB_APPLICATION_TOKEN,
    namespace=ASTRA_DB_KEYSPACE
)

# 4. Query
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatGroq

llm = ChatGroq(groq_api_key="your-groq-key", model_name="llama3-8b-8192")

qa = RetrievalQA.from_chain_type(llm=llm, retriever=vectordb.as_retriever())
answer = qa.run("Which phone is best under 10000?")
print(f"Bot: {response}\n")
