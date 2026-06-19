# 📄 Research Paper Deep Dive - RAG System

## Overview

Research Paper Deep Dive is a Retrieval-Augmented Generation (RAG) application that enables users to interact with and analyze research papers using Large Language Models.

The system processes multiple research papers, generates embeddings, stores them in a Qdrant Vector Database, retrieves relevant document chunks, and generates accurate answers using Google's Gemini model.

---

## Research Domain

This project focuses on **Large Language Models (LLMs) and Instruction Tuning**.

The paper collection covers:

* Instruction Tuning
* Reinforcement Learning from Human Feedback (RLHF)
* LLM Alignment
* Fine-Tuning Techniques
* Efficient Transformer Architectures

A collection of 15 research papers was used for knowledge retrieval and question answering.

---

## Features

* Multi-PDF Research Paper Processing
* Docling-based Document Parsing
* Intelligent Text Chunking
* Semantic Search with HuggingFace Embeddings
* Qdrant Vector Database Integration
* Retrieval-Augmented Generation (RAG)
* Gemini-powered Question Answering
* Source Document Traceability
* Interactive Streamlit Interface

---

## Tech Stack

* Python
* Streamlit
* LangChain
* Docling
* HuggingFace Embeddings
* Qdrant Vector Database
* Google Gemini
* Sentence Transformers

---

## System Architecture

Research Papers (PDFs)

→ Docling Loader

→ Text Chunking

→ HuggingFace Embeddings

→ Qdrant Vector Store

→ Retriever

→ Gemini LLM

→ Answer Generation

---

## Project Workflow

1. Load research papers from the dataset.
2. Extract and process text using Docling.
3. Split documents into smaller chunks.
4. Generate vector embeddings using Sentence Transformers.
5. Store embeddings in Qdrant.
6. Retrieve the most relevant chunks based on user queries.
7. Generate answers using Gemini with retrieved context.
8. Display responses along with source references.

---

## Installation

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

---

## Example Questions

* What is Instruction Tuning?
* What is RLHF and why is it important?
* Compare Instruction Tuning and Fine-Tuning.
* How do Transformer architectures improve model performance?
* What are the key findings from the research papers?

---

## Project Structure

```text
Research_Paper_Deep_Dive-_RAG_System/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
└── data/
    └── raw/
        ├── research_paper_1.pdf
        ├── research_paper_2.pdf
        └── ...
```

---

## Future Improvements

* Chat History Support
* Research Paper Summarization
* Citation Generation
* Multi-Collection Search
* Hybrid Retrieval (Dense + Sparse)
* Research Trend Analysis

---

## Author

**Shivaji**
B.Tech Graduate | Generative AI & Data Science Enthusiast


