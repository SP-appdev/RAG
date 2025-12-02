# Clinical RAG Assistant

A simple **Retrieval-Augmented Generation (RAG)** system that answers questions from a JSON dataset using **SambaNova LLM** and vector search. The system reads JSON data, chunks it, embeds the chunks, stores them in a **FAISS** vector database, and retrieves relevant context to answer user queries.

---

## Features

- Load clinic information from a JSON file.
- Chunk long text for better retrieval.
- Embed text chunks using **MiniLM embeddings**.
- Store embeddings in a **FAISS** vector database for fast similarity search.
- Retrieve relevant chunks to answer questions using **SambaNova LLM**.
- Command-line interface (CLI) for interactive question-answering.

---

## Requirements

```text
python >= 3.10
faiss
numpy
langchain
langchain-community
sentence-transformers
openai
