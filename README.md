# Research Paper Deep Dive using RAG

## Overview

Research Paper Deep Dive is a Retrieval-Augmented Generation (RAG) application that enables users to query and analyze multiple research papers using Large Language Models.

The system processes research papers, stores embeddings in Qdrant Vector Database, retrieves relevant document chunks, and generates context-aware answers using Google Gemini.

## Features

* Multi-PDF research paper ingestion
* Document parsing using Docling
* Semantic search with HuggingFace embeddings
* Qdrant Vector Database integration
* Retrieval-Augmented Generation (RAG)
* Google Gemini-powered question answering
* Streamlit-based user interface
* Source document tracking

## Tech Stack

* Python
* Streamlit
* LangChain
* Docling
* HuggingFace Embeddings
* Qdrant
* Google Gemini

## Workflow

PDF Papers
→ Docling Loader
→ Text Chunking
→ HuggingFace Embeddings
→ Qdrant Vector Store
→ Retriever
→ Gemini LLM
→ Answer Generation

## Installation

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Example Questions

* What is instruction tuning?
* What is RLHF?
* Compare Llama 2 and Gemma.
* Summarize the key contributions of the papers.

## Author

Shivaji
