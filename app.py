import os
import streamlit as st

from langchain_docling.loader import DoclingLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

# Page Config

st.set_page_config(
page_title="Research Paper Deep Dive",
page_icon="📄"
)

st.title("📄 Research Paper Deep Dive")

st.markdown("""

### Large Language Models & Instruction Tuning

Ask questions across 15 research papers covering:

* Instruction Tuning
* RLHF (Reinforcement Learning from Human Feedback)
* LLM Alignment
* Fine-Tuning Techniques
* Efficient Transformer Architectures
  """)

# API Key

GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]

@st.cache_resource
def load_rag():
 docs = []

 for file in os.listdir("data/raw"):
    if file.endswith(".pdf"):
        docs.extend(
            DoclingLoader(
                f"data/raw/{file}"
            ).load()
        )

 chunks = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=50
 ).split_documents(docs)

 embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
 )

 vectorstore = QdrantVectorStore.from_documents(
    documents=chunks,
    embedding=embeddings,
    path="./qdrant/dense",
    collection_name="research_papers"
 )

 retriever = vectorstore.as_retriever(
    search_kwargs={"k": 5}
 )

 llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0
 )

 prompt = ChatPromptTemplate.from_template("""
 Answer the question using only the provided context.

 Context:
 {context}

 Question:
 {question}

 If the answer is not found in the context,
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


query = st.text_input(
"Ask a question about the research papers"
)

if st.button("Submit") and query:
 answer, docs = ask_question(query)
 st.subheader("Answer")
 st.write(answer)
with st.expander("Sources"):
 for doc in docs:
  st.write(doc.metadata)

