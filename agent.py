# ============================================================
# agent.py — StyleCart Agentic AI Capstone (Gemini 2.5 Version)
# ============================================================

import os
from datetime import datetime
from typing import TypedDict, List

import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from dotenv import load_dotenv
load_dotenv()


# ── Configuration ────────────────────────────────────────────
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

MODEL_NAME = "gemini-2.5-flash"

FAITHFULNESS_THRESHOLD = 0.7
MAX_EVAL_RETRIES = 2
SLIDING_WINDOW = 6


# ── State ─────────────────────────────────────────────────────
class CapstoneState(TypedDict):
    question: str
    messages: List[dict]
    route: str
    retrieved: str
    sources: List[str]
    tool_result: str
    answer: str
    faithfulness: float
    eval_retries: int
    customer_name: str


# ── Resource initialisation ───────────────────────────────────
print("[agent.py] Loading resources...")

model = genai.GenerativeModel(MODEL_NAME)

embedder = SentenceTransformer("all-MiniLM-L6-v2")

client = chromadb.Client(
    Settings(persist_directory="./chroma_db")
)

try:
    client.delete_collection("stylecart_kb")
except:
    pass

collection = client.create_collection("stylecart_kb")


_documents = [
    {
        "id": "doc_001",
        "topic": "Return Policy",
        "text": "StyleCart accepts product returns within 7 days of delivery..."
    },
    {
        "id": "doc_002",
        "topic": "Shipping and Delivery",
        "text": "StyleCart ships to all major cities across India..."
    },
    {
        "id": "doc_003",
        "topic": "Payment Methods",
        "text": "StyleCart accepts UPI, credit cards, debit cards..."
    }
]


_texts = [d["text"] for d in _documents]
_embeddings = embedder.encode(_texts).tolist()

collection.add(
    documents=_texts,
    embeddings=_embeddings,
    ids=[d["id"] for d in _documents],
    metadatas=[{"topic": d["topic"]} for d in _documents]
)

print(f"[agent.py] ChromaDB ready: {collection.count()} documents.")


# ── Node functions ────────────────────────────────────────────
def memory_node(state: CapstoneState) -> dict:
    messages = state.get("messages", []) + [{"role": "user", "content": state["question"]}]
    messages = messages[-SLIDING_WINDOW:]

    customer_name = state.get("customer_name", "")
    if "my name is" in state["question"].lower():
        parts = state["question"].lower().split("my name is")
        if len(parts) > 1:
            customer_name = parts[1].strip().split()[0].capitalize()

    return {
        "messages": messages,
        "customer_name": customer_name,
        "eval_retries": 0
    }


def router_node(state: CapstoneState) -> dict:
    history = "\n".join(
        [f"{m['role'].upper()}: {m['content']}" for m in state.get("messages", [])[-4:]]
    )

    prompt = f"""Router for StyleCart support. Reply with ONE word: retrieve, tool, or memory_only.
History: {history}
Question: {state['question']}
Route:"""

    response = model.generate_content(prompt)
    route = response.text.strip().lower()

    if route not in ["retrieve", "tool", "memory_only"]:
        route = "retrieve"

    return {"route": route}


def retrieval_node(state: CapstoneState) -> dict:
    qe = embedder.encode([state["question"]]).tolist()
    results = collection.query(query_embeddings=qe, n_results=2)

    context_parts, sources = [], []

    for chunk, meta in zip(results["documents"][0], results["metadatas"][0]):
        context_parts.append(f"[{meta['topic']}]\n{chunk}")
        sources.append(meta["topic"])

    return {
        "retrieved": "\n\n".join(context_parts),
        "sources": sources
    }


def tool_node(state: CapstoneState) -> dict:
    now = datetime.now()
    return {
        "tool_result": f"Today is {now.strftime('%A, %d %B %Y')}",
        "retrieved": "",
        "sources": []
    }


def answer_node(state: CapstoneState) -> dict:
    ctx = state.get("retrieved", "") or state.get("tool_result", "")

    prompt = f"""
Answer ONLY from context.

Context:
{ctx}

Question:
{state['question']}
"""

    response = model.generate_content(prompt)

    return {
        "answer": response.text.strip()
    }


def save_node(state: CapstoneState) -> dict:
    return {
        "messages": state.get("messages", []) + [
            {"role": "assistant", "content": state.get("answer", "")}
        ]
    }


# ── Graph assembly ────────────────────────────────────────────
g = StateGraph(CapstoneState)

g.add_node("memory", memory_node)
g.add_node("router", router_node)
g.add_node("retrieve", retrieval_node)
g.add_node("tool", tool_node)
g.add_node("answer", answer_node)
g.add_node("save", save_node)

g.set_entry_point("memory")

g.add_edge("memory", "router")
g.add_edge("retrieve", "answer")
g.add_edge("tool", "answer")
g.add_edge("answer", "save")
g.add_edge("save", END)

g.add_conditional_edges(
    "router",
    lambda state: "tool" if state["route"] == "tool" else "retrieve",
    {
        "retrieve": "retrieve",
        "tool": "tool"
    }
)

app = g.compile(checkpointer=MemorySaver())

print("[agent.py] Graph compiled successfully.")


# ── Public API ────────────────────────────────────────────────
def ask(question: str, thread_id: str = "default") -> dict:
    config = {"configurable": {"thread_id": thread_id}}
    return app.invoke({"question": question}, config=config)