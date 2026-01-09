import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader
import docx
import pandas as pd
import json

# --- Configuration ---
FAISS_BASE_PATH = "faiss_indexes"
os.makedirs(FAISS_BASE_PATH, exist_ok=True)

# Embedding model (compact + fast)
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# --- Helper paths ---
def _index_path(conversation_id):
    return os.path.join(FAISS_BASE_PATH, f"conversation_{conversation_id}.index")

def _chunks_path(conversation_id):
    return os.path.join(FAISS_BASE_PATH, f"conversation_{conversation_id}_chunks.json")

# --- Extract text from uploaded files ---
def extract_text_from_file(uploaded_file):
    name = uploaded_file.name.lower()
    text = ""

    try:
        if name.endswith(".pdf"):
            reader = PdfReader(uploaded_file)
            text = "\n".join([page.extract_text() or "" for page in reader.pages])

        elif name.endswith(".docx") or name.endswith(".doc"):
            doc = docx.Document(uploaded_file)
            text = "\n".join([para.text for para in doc.paragraphs])

        elif name.endswith(".xls") or name.endswith(".xlsx"):
            xls = pd.read_excel(uploaded_file, sheet_name=None, dtype=str)
            parts = []
            for sheet_name, df in xls.items():
                parts.append(f"\n\nSheet: {sheet_name}\n{df.fillna('').to_string(index=False)}")
            text = "\n".join(parts)

        elif name.endswith(".csv"):
            df = pd.read_csv(uploaded_file, dtype=str, keep_default_na=False)
            text = df.to_string(index=False)

        else:
            text = uploaded_file.read().decode(errors="ignore")

    except Exception as e:
        text = f"[Error reading file: {e}]"

    return text.strip()


# --- Split text into chunks ---
def chunk_text(text, chunk_size=500):
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]


# --- FAISS index management ---
def load_index(conversation_id):
    path = _index_path(conversation_id)
    if os.path.exists(path):
        return faiss.read_index(path)
    dim = embedding_model.get_sentence_embedding_dimension()
    return faiss.IndexFlatL2(dim)

def save_index(index, conversation_id):
    faiss.write_index(index, _index_path(conversation_id))


# --- Add document content ---
def add_document(conversation_id, text):
    text = text.strip()
    if not text:
        return []

    chunks = chunk_text(text)
    vectors = embedding_model.encode(chunks)
    
    # Load or create index
    index = load_index(conversation_id)
    index.add(np.array(vectors).astype("float32"))
    save_index(index, conversation_id)

    # Save chunks to JSON for retrieval
    chunks_path = _chunks_path(conversation_id)
    if os.path.exists(chunks_path):
        with open(chunks_path, "r", encoding="utf-8") as f:
            existing_chunks = json.load(f)
    else:
        existing_chunks = []

    existing_chunks.extend(chunks)
    with open(chunks_path, "w", encoding="utf-8") as f:
        json.dump(existing_chunks, f, ensure_ascii=False, indent=2)

    print(f"✅ Added {len(chunks)} chunks to conversation {conversation_id}")
    return chunks


# --- Retrieve relevant chunks ---
def retrieve_relevant_chunks(conversation_id, query, top_k=3):
    index_path = _index_path(conversation_id)
    chunks_path = _chunks_path(conversation_id)

    if not os.path.exists(index_path) or not os.path.exists(chunks_path):
        return []

    index = faiss.read_index(index_path)
    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    query_vector = embedding_model.encode([query]).astype("float32")
    distances, indices = index.search(query_vector, top_k)

    relevant = []
    for idx in indices[0]:
        if 0 <= idx < len(chunks):
            relevant.append(chunks[idx])
    return relevant


# --- Delete FAISS memory for conversation ---
def delete_index(conversation_id):
    for path in [_index_path(conversation_id), _chunks_path(conversation_id)]:
        if os.path.exists(path):
            os.remove(path)
    print(f"🗑️ Deleted FAISS memory for conversation {conversation_id}")

