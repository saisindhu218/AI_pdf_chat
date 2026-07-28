<<<<<<< HEAD
#  AI PDF Chat (RAG)
=======
# 📄 AI PDF Chat (RAG)
>>>>>>> 68bf111 (mostly done)

> Intelligent PDF Question Answering using Retrieval-Augmented Generation (RAG)

AI PDF Chat is a Retrieval-Augmented Generation (RAG) application that enables users to upload PDF documents and ask natural language questions. The application retrieves the most relevant document sections using semantic search and generates context-aware answers with page-level citations.

The project supports both **Groq-hosted LLMs** for high-quality responses and a **fully local inference pipeline** using Flan-T5, allowing it to operate with or without external APIs.

<<<<<<< HEAD


## Features

-  Upload and process PDF documents
-  Semantic search using Sentence Transformers
-  Retrieval-Augmented Generation (RAG)
-  FAISS vector indexing for fast similarity search
-  Context-aware question answering
-  Page-level source citations
-  Dual inference support
  - Groq (Llama 3.1 / 3.3)
  - Local Flan-T5 fallback
-  API key managed securely using environment variables
-  Deployable on Streamlit Community Cloud and Hugging Face Spaces



##  System Architecture
=======
---

## 🚀 Features

- 📄 Upload and process PDF documents
- 🔍 Semantic search using Sentence Transformers
- 🧠 Retrieval-Augmented Generation (RAG)
- ⚡ FAISS vector indexing for fast similarity search
- 💬 Context-aware question answering
- 📑 Page-level source citations
- 🤖 Dual inference support
  - Groq (Llama 3.1 / 3.3)
  - Local Flan-T5 fallback
- 🔐 API key managed securely using environment variables
- 🌐 Deployable on Streamlit Community Cloud and Hugging Face Spaces

---

## 🏗️ System Architecture
>>>>>>> 68bf111 (mostly done)

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
        |  Groq (Llama 3.x)             |
        |            OR                 |
        |  Local Flan-T5                |
        +-------------------------------+
                           |
                           ▼
         Grounded Answer + Page Citation
```

<<<<<<< HEAD


##  Technology Stack

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



##  Workflow

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



##  Project Structure
=======
---

## ⚙️ Technology Stack

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

## 🔄 Workflow

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

## 📂 Project Structure
>>>>>>> 68bf111 (mostly done)

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

<<<<<<< HEAD


##  Installation
=======
---

## 🛠️ Installation
>>>>>>> 68bf111 (mostly done)

Clone the repository

```bash
git clone https://github.com/yourusername/AI_PDF_Chat.git

cd AI_PDF_Chat
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
python -m streamlit run app.py
```

<<<<<<< HEAD

##  Configuration
=======
---

## 🔑 Configuration
>>>>>>> 68bf111 (mostly done)

Create a `.env` file in the project root.

```env
GROQ_API_KEY=your_api_key_here
```

If no API key is provided, the application automatically switches to the local Flan-T5 model.

<<<<<<< HEAD


##  Example Questions

- Summarize this document.
- Explain the main projects described.
- What technologies are mentioned?
- What responsibilities are listed?
- What skills are highlighted?
- Which page contains information about machine learning?



##  Key Highlights

- Retrieval-Augmented Generation (RAG)
- Semantic Search
- Dense Vector Embeddings
- FAISS Similarity Search
- Local + Cloud Inference
- Source Grounding
- Page-Level Citations
- Environment Variable Based Secret Management



##  Deployment

=======
---

## 💡 Example Questions

- Summarize this document.
- Explain the main projects described.
- What technologies are mentioned?
- What responsibilities are listed?
- What skills are highlighted?
- Which page contains information about machine learning?

---

## 📊 Key Highlights

- Retrieval-Augmented Generation (RAG)
- Semantic Search
- Dense Vector Embeddings
- FAISS Similarity Search
- Local + Cloud Inference
- Source Grounding
- Page-Level Citations
- Environment Variable Based Secret Management

---

## 🚀 Deployment

>>>>>>> 68bf111 (mostly done)
The application can be deployed on:

- Streamlit Community Cloud
- Hugging Face Spaces

For production deployments, store the Groq API key using the platform's Secrets Manager instead of hardcoding it.

<<<<<<< HEAD

##  Future Enhancements
=======
---

## 🔮 Future Enhancements
>>>>>>> 68bf111 (mostly done)

- Multi-document RAG
- Persistent Vector Database
- OCR support for scanned PDFs
- Chat history
- Metadata filtering
- Hybrid keyword + semantic search
- Multi-user authentication
- Conversation memory

<<<<<<< HEAD


##  Interview Discussion Points
=======
---

## 🎯 Interview Discussion Points
>>>>>>> 68bf111 (mostly done)

This project demonstrates practical implementation of Retrieval-Augmented Generation (RAG) by combining semantic retrieval with large language models to answer questions grounded in document content.

Key engineering concepts demonstrated include:

- Document preprocessing
- Text chunking strategies
- Embedding generation
- Vector similarity search
- Retrieval pipelines
- Context-aware response generation
- Local vs Cloud inference trade-offs
- Secure API key management

<<<<<<< HEAD


##  License

This project is licensed under the MIT License.



##  Author

**Sai Sindhu Rachabattuni**

Java Developer | AI & Full Stack Developer

GitHub: https://github.com/saisindhu218

LinkedIn: https://linkedin.com/in/sai-sindhu-rachabattuni-b241a52b6
=======
---

## 📜 License

This project is licensed under the MIT License.

---

## 👨‍💻 Author

**Sai Sindhu Rachabattuni**


GitHub: 

LinkedIn: 
>>>>>>> 68bf111 (mostly done)
