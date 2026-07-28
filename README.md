# AI PDF Chat (RAG)

> Intelligent PDF Question Answering using Retrieval-Augmented Generation (RAG)

AI PDF Chat is a Retrieval-Augmented Generation (RAG) application that enables users to upload PDF documents and ask natural language questions. The application retrieves the most relevant document sections using semantic search and generates context-aware answers with page-level citations.

The project supports both **Groq-hosted LLMs** for high-quality responses and a **fully local inference pipeline** using Flan-T5, allowing it to operate with or without external APIs.

---

## Features

- Upload and process PDF documents
- Semantic search using Sentence Transformers
- Retrieval-Augmented Generation (RAG)
- FAISS vector indexing for fast similarity search
- Context-aware question answering
- Page-level source citations
- Dual inference support:
  - Groq (Llama 3.1 / 3.3)
  - Local Flan-T5 fallback
- Secure API key management using environment variables
- Deployable on Streamlit Community Cloud and Hugging Face Spaces

---

## System Architecture

```text
                 +--------------------+
                 |    PDF Upload      |
                 +---------+----------+
                           |
                           ▼
                Text Extraction (PyMuPDF)
                           |
                           ▼
               Intelligent Text Chunking
                           |
                           ▼
          Sentence Embeddings (MiniLM-L6-v2)
                           |
                           ▼
               FAISS Vector Database
                           |
                           ▼
              Top-K Semantic Retrieval
                           |
                           ▼
        +-------------------------------+
        |      Response Generation      |
        |-------------------------------|
        |  Groq (Llama 3.1 / 3.3)        |
        |             OR                |
        |  Local Flan-T5                |
        +-------------------------------+
                           |
                           ▼
         Grounded Answer + Page Citation
```

---

## Technology Stack

### Frontend

- Streamlit

### Backend

- Python

### AI & NLP

- Sentence Transformers
- all-MiniLM-L6-v2
- Flan-T5
- Groq API

### Vector Search

- FAISS

### PDF Processing

- PyMuPDF

---

## Workflow

1. Upload a PDF document.
2. Extract text using PyMuPDF.
3. Split the content into semantic chunks.
4. Generate embeddings using MiniLM.
5. Store embeddings in a FAISS vector index.
6. Retrieve the most relevant chunks based on the user's query.
7. Generate a grounded answer using:
   - Groq LLM (preferred), or
   - Local Flan-T5 model.
8. Display the answer along with the source page citation.

---

## Project Structure

```text
AI_PDF_Chat/
│
├── app.py
├── rag_utils.py
├── requirements.txt
├── README.md
├── .env.example
└── assets/
```

---

## Installation

### Clone the repository

```bash
git clone https://github.com/saisindhu218/AI_pdf_chat.git
cd AI_pdf_chat
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the application

```bash
streamlit run app.py
```

---

## Configuration

Create a `.env` file in the project root.

```env
GROQ_API_KEY=your_groq_api_key
```

If no API key is provided, the application automatically switches to the local Flan-T5 model.

---

## Example Questions

- Summarize this document.
- Explain the main topics discussed.
- What technologies are mentioned?
- What responsibilities are listed?
- What skills are highlighted?
- Which page discusses machine learning?

---

## Key Highlights

- Retrieval-Augmented Generation (RAG)
- Semantic Search
- Dense Vector Embeddings
- FAISS Similarity Search
- Local and Cloud Inference
- Source Grounding
- Page-Level Citations
- Environment Variable-Based Secret Management

---

## Deployment

The application can be deployed on:

- Streamlit Community Cloud
- Hugging Face Spaces

For production deployments, store the Groq API key using the platform's Secrets Manager instead of hardcoding it.

---

## Future Enhancements

- Multi-document RAG
- Persistent Vector Database
- OCR support for scanned PDFs
- Chat history
- Metadata filtering
- Hybrid keyword and semantic search
- Multi-user authentication
- Conversation memory

---

## Interview Discussion Points

This project demonstrates the practical implementation of Retrieval-Augmented Generation (RAG) by combining semantic retrieval with large language models to generate answers grounded in document content.

Key engineering concepts include:

- Document preprocessing
- Text chunking strategies
- Embedding generation
- Vector similarity search
- Retrieval pipelines
- Context-aware response generation
- Local vs. Cloud inference trade-offs
- Secure API key management

---

## License

This project is licensed under the MIT License.

---

## Author

**Sai Sindhu Rachabattuni**

Entry-Level Software Engineer | Java & Full-Stack Developer

**GitHub:** https://github.com/saisindhu218

**LinkedIn:** https://www.linkedin.com/in/sai-sindhu-rachabattuni-b241a52b6/
