# Hierarchical - RAG: Multimodal Hierarchical RAG for Financial Reports

A multimodal Retrieval-Augmented Generation (RAG) pipeline designed to extract, orchestrate, and reason over complex financial tables and dense text in 10-K PDFs. 

Built with **Python**, **Streamlit**, and **Gemini 2.5 Flash**, this architecture solves the notorious "table hallucination" problem found in traditional flat-text RAG systems.

## Core Architecture

Traditional RAG systems parse PDFs into flat text, destroying the structure of complex financial tables. **HierFinRAG** solves this using a 3-tier architecture:

1. **Structural PDF Parsing (PyMuPDF):** 
   Documents are parsed into a Graph structure. Pages become **Parent Nodes**, while paragraphs become **Child Text Nodes** and complex tables/charts are extracted as **Child Image Nodes**.
2. **Multi-Vector Graph Store:** 
   Text nodes are embedded into a dense vector space, but they maintain a mathematical bridge to their Parent Page.
3. **Multimodal Parent Retrieval:** 
   When a user asks a question, the hybrid search engine finds the most relevant child paragraph. The orchestration layer then traverses the graph to retrieve the *entire Parent Page*, including the raw visual images of any tables on that page. This multimodal payload is passed directly to **Gemini 2.5 Flash** for visual reasoning.

## LLM-as-a-Judge Evaluation Framework

This repository includes a custom evaluation pipeline (`benchmark.py`) that stress-tests the model using an LLM-as-a-Judge approach. 

The benchmark calculates:
- **Mean Reciprocal Rank (MRR)** & **Context Recall:** Mathematically scoring the Vector Store's retrieval accuracy against a Golden Dataset.
- **Faithfulness:** An independent Gemini Judge verifies that the generated answer strictly relies on the retrieved context without hallucinating external information.
- **Answer Relevance:** The Judge scores how directly the generated answer addresses the user's prompt.

## Getting Started

1. Clone the repository and set up a virtual environment.
2. Install the required dependencies (Streamlit, PyMuPDF, Google Generative AI, Sentence-Transformers).
3. Set your `GEMINI_API_KEY` in a `.env` file.
4. Run the UI:
```bash
streamlit run streamlit.py
```
5. Upload a financial PDF (e.g., a Microsoft 10-K) and ask questions about the tables!


