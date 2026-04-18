# 🛍️ StyleCart AI Support Agent

An end-to-end **AI-powered customer support chatbot** built using **LangGraph, Gemini (Google Generative AI), ChromaDB, and Streamlit**.

This project simulates a real-world **e-commerce support assistant** that can answer queries related to returns, shipping, payments, and more — using a **Retrieval-Augmented Generation (RAG)** pipeline with evaluation for answer faithfulness.

---

## 🚀 Features

* 🤖 **LLM-Powered Chatbot** using Gemini (2.5 Flash / 1.5 Flash)
* 🔍 **RAG Pipeline** (Retrieval-Augmented Generation)
* 🧠 **Conversation Memory** with sliding window
* 📚 **Vector Database** using ChromaDB
* 📊 **Faithfulness Evaluation Loop** (self-correction)
* 🔀 **Smart Routing**

  * Retrieve from knowledge base
  * Use tool (date/time)
  * Answer from memory
* 💬 **Streamlit Chat UI**
* ⚡ **Cached Model Loading** for performance

---

## 🏗️ Architecture

```
User Query
   ↓
Memory Node (store conversation)
   ↓
Router Node (decide action)
   ↓
 ┌───────────────┬───────────────┬───────────────┐
 ↓               ↓               ↓
Retrieve       Tool          Memory Only
 ↓               ↓               ↓
        → Answer Generation Node →
                   ↓
            Evaluation Node
                   ↓
         Retry OR Save Response
```

---

## 🧰 Tech Stack

| Category   | Tools Used                                |
| ---------- | ----------------------------------------- |
| LLM        | Gemini 2.5 Flash (Google Generative AI)   |
| Framework  | LangGraph + LangChain                     |
| Embeddings | SentenceTransformers (`all-MiniLM-L6-v2`) |
| Vector DB  | ChromaDB                                  |
| UI         | Streamlit                                 |
| Language   | Python                                    |

---

## 📦 Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/stylecart-ai-agent.git
cd stylecart-ai-agent
```

### 2. Create virtual environment (recommended)

```bash
python -m venv venv
venv\Scripts\activate   # Windows
source venv/bin/activate # Mac/Linux
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install streamlit langchain langgraph langchain-google-genai sentence-transformers chromadb
```

---

## 🔑 API Key Setup

Set your Google API key:

### Windows:

```bash
setx GOOGLE_API_KEY "your_api_key_here"
```

### Mac/Linux:

```bash
export GOOGLE_API_KEY="your_api_key_here"
```

Restart terminal after setting.

---

## ▶️ Run the App

```bash
streamlit run capstone_streamlit.py
```

---

## 💡 Example Questions

Try asking:

* "What is the return policy?"
* "How long does delivery take?"
* "Do you support COD?"
* "What payment methods are available?"
* "What is today's date?"

---

## 🧠 How It Works

### 1. Retrieval

User query is embedded and matched against stored documents using **ChromaDB**.

### 2. Generation

Gemini generates answers **only from retrieved context**.

### 3. Evaluation (Key Feature 🚀)

* The system checks if the answer is **faithful to the context**
* If not → it retries generation (self-correcting loop)

### 4. Memory

Maintains last **6 messages** to preserve conversation flow.

---

## 📁 Project Structure

```
.
├── capstone_streamlit.py   # Main Streamlit app
├── README.md               # Project documentation
├── requirements.txt        # Dependencies
```

---

## ⚠️ Limitations

* Free tier of Gemini has **rate limits**
* Knowledge base is static (can be expanded)
* No authentication / user accounts
* Evaluation is basic (can be improved with RAGAS)

---

## 🔮 Future Improvements

* 🔁 Add **RAGAS evaluation metrics**
* 📊 Dashboard for analytics
* 🌐 Deploy on Streamlit Cloud / AWS
* 🧾 Add real database (orders, users)
* 🔎 Improve retrieval with reranking
* 🗂️ Dynamic document ingestion

---

## 🏆 Learning Outcomes

This project demonstrates:

* Building a **multi-node AI system using LangGraph**
* Implementing **RAG from scratch**
* Integrating **LLMs into real applications**
* Designing **evaluation loops for reliability**
* Creating a **production-style AI assistant**

---

## 👩‍💻 Author

**Shreyoshi Ghosh**

---

## ⭐ If you like this project

Give it a ⭐ on GitHub and feel free to contribute!

---
