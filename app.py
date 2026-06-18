import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore, RetrievalMode
from qdrant_client import QdrantClient


load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
QDRANT_PATH = Path("./qdrant/dense")
COLLECTION_NAME = "research_papers"


st.set_page_config(
    page_title="Research Paper Deep Dive",
    page_icon="📄",
)

st.title("📄 Research Paper Deep Dive")

st.markdown(
    """
### Large Language Models & Instruction Tuning

Ask questions across 15 research papers covering:
- Instruction Tuning
- Reinforcement Learning from Human Feedback (RLHF)
- LLM Alignment
- Fine-Tuning Techniques
- Efficient Transformer Architectures
"""
)

st.write("Ask questions about your research papers")


@st.cache_resource
def load_rag():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    client = QdrantClient(
        path=str(QDRANT_PATH)
    )

    vectorstore = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings,
        retrieval_mode=RetrievalMode.DENSE,
    )

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 5}
    )

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=GOOGLE_API_KEY,
        temperature=0,
    )

    prompt = ChatPromptTemplate.from_template(
        """
Answer the question using only the provided context.

Context:
{context}

Question:
{question}

If the answer is not available in the context,
say "I could not find this in the provided papers."
"""
    )

    chain = prompt | llm

    return retriever, chain


def ask_question(query):
    docs = retriever.invoke(query)
    context = "\n\n".join(doc.page_content for doc in docs)

    response = chain.invoke(
        {
            "context": context,
            "question": query,
        }
    )

    return response.content, docs


if not GOOGLE_API_KEY:
    st.error("Add GOOGLE_API_KEY to your .env file before asking questions.")
    st.stop()

if not QDRANT_PATH.exists():
    st.error(f"Qdrant database not found at {QDRANT_PATH}.")
    st.stop()


retriever, chain = load_rag()

query = st.text_input("Ask a question")

if st.button("Submit") and query:
    answer, docs = ask_question(query)

    st.subheader("Answer")
    st.write(answer)

    with st.expander("Sources"):
        for doc in docs:
            st.write(doc.metadata)
