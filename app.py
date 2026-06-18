import os
import streamlit as st
from dotenv import load_dotenv

from langchain_docling.loader import DoclingLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore, RetrievalMode
from qdrant_client import QdrantClient
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

# Load API Key

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Page Config

st.set_page_config(
page_title="Research Paper Deep Dive",
page_icon="📄"
)

st.title("📄 Research Paper Deep Dive")
st.write("Ask questions about your research papers")

@st.cache_resource
def load_rag():


 # Embeddings
 embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
 )

# Load Existing Qdrant DB
 client = QdrantClient(
    path="./qdrant/dense"
 )

 vectorstore = QdrantVectorStore(
    client=client,
    collection_name="research_papers",
    embedding=embeddings,
    retrieval_mode=RetrievalMode.DENSE
 )

 retriever = vectorstore.as_retriever(
    search_kwargs={"k": 5}
 )

# Gemini
 llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0
 )

# Prompt
 prompt = ChatPromptTemplate.from_template("""
Answer the question using only the provided context.

Context:
{context}

Question:
{question}

If the answer is not available in the context,
say "I could not find this in the provided papers."
""")

 chain = prompt | llm

 return retriever, chain


retriever, chain = load_rag()

def ask_question(query):
   docs = retriever.invoke(query)
   context = "\n\n".join(doc.page_content for doc in docs)
   response = chain.invoke({
    "context": context,
    "question": query
}) 
   return response.content, docs


# UI

query = st.text_input(
"Ask a question"
)

if st.button("Submit") and query:
   answer, docs = ask_question(query)
   st.subheader("Answer")
   st.write(answer)
   with st.expander("Sources"):
     for doc in docs:
       st.write(doc.metadata)

