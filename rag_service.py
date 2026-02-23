"""
RAG Service - Document-based Q&A for Brain Buddy
Built from scratch using OpenAI embeddings + ChromaDB
"""

import os
import json
import uuid
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional

import chromadb
from chromadb.config import Settings

# Text extraction libraries
import PyPDF2
from docx import Document as DocxDocument


# --- Configuration ---
DATA_DIR = Path(__file__).parent / "student_data"
DATA_DIR.mkdir(exist_ok=True)

# ChromaDB persistent client
chroma_client = chromadb.PersistentClient(path=str(DATA_DIR / "chroma_db"))

# Metadata store for document info (simple JSON file)
METADATA_FILE = DATA_DIR / "documents_metadata.json"


def _load_metadata() -> Dict:
    """Load document metadata from JSON file."""
    if METADATA_FILE.exists():
        with open(METADATA_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_metadata(metadata: Dict):
    """Save document metadata to JSON file."""
    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


# =============================================
# 1. TEXT EXTRACTION
# =============================================

def extract_text_from_pdf(file_path: str) -> str:
    """Extract text from a PDF file."""
    reader = PyPDF2.PdfReader(file_path)
    text_parts = []
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text_parts.append(page_text)
    return "\n\n".join(text_parts)


def extract_text_from_docx(file_path: str) -> str:
    """Extract text from a DOCX file."""
    doc = DocxDocument(file_path)
    text_parts = []
    for para in doc.paragraphs:
        if para.text.strip():
            text_parts.append(para.text)
    return "\n\n".join(text_parts)


def extract_text_from_txt(file_path: str) -> str:
    """Extract text from a TXT file."""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def extract_text(file_path: str, file_type: str) -> str:
    """Extract text from a file based on its type."""
    extractors = {
        "pdf": extract_text_from_pdf,
        "docx": extract_text_from_docx,
        "txt": extract_text_from_txt,
    }
    
    ext = file_type.lower()
    if ext not in extractors:
        raise ValueError(f"Unsupported file type: {ext}. Supported: {list(extractors.keys())}")
    
    return extractors[ext](file_path)


# =============================================
# 2. TEXT CHUNKING
# =============================================

def chunk_text(text: str, chunk_size: int = 400, overlap: int = 50) -> List[str]:
    """
    Split text into overlapping word-based chunks.
    
    Args:
        text: The full document text
        chunk_size: Number of words per chunk
        overlap: Number of overlapping words between chunks
    
    Returns:
        List of text chunks
    """
    words = text.split()
    
    if len(words) <= chunk_size:
        return [text.strip()] if text.strip() else []
    
    chunks = []
    start = 0
    
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        if chunk.strip():
            chunks.append(chunk.strip())
        start += chunk_size - overlap
    
    return chunks


# =============================================
# 3. EMBEDDING & STORAGE
# =============================================

def store_document(doc_id: str, filename: str, chunks: List[str], openai_client) -> Dict:
    """
    Embed text chunks and store them in ChromaDB.
    
    Returns:
        Document metadata dict
    """
    collection = chroma_client.get_or_create_collection(
        name=doc_id,
        metadata={"hnsw:space": "cosine"}
    )
    
    # Embed all chunks in a single API call (batch)
    batch_size = 100  # OpenAI allows up to 2048
    all_ids = []
    all_embeddings = []
    all_documents = []
    all_metadatas = []
    
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        
        response = openai_client.embeddings.create(
            input=batch,
            model="text-embedding-3-small"
        )
        
        for j, embedding_data in enumerate(response.data):
            idx = i + j
            all_ids.append(f"{doc_id}_chunk_{idx}")
            all_embeddings.append(embedding_data.embedding)
            all_documents.append(batch[j])
            all_metadatas.append({"chunk_index": idx, "doc_id": doc_id})
    
    # Add to ChromaDB
    collection.add(
        ids=all_ids,
        embeddings=all_embeddings,
        documents=all_documents,
        metadatas=all_metadatas
    )
    
    # Save metadata
    metadata = _load_metadata()
    doc_meta = {
        "id": doc_id,
        "filename": filename,
        "num_chunks": len(chunks),
        "total_words": sum(len(c.split()) for c in chunks),
        "uploaded_at": datetime.now().isoformat(),
    }
    metadata[doc_id] = doc_meta
    _save_metadata(metadata)
    
    return doc_meta


# =============================================
# 4. QUERY (RAG)
# =============================================

def query_document(doc_id: str, question: str, openai_client, n_results: int = 4) -> Dict:
    """
    Ask a question about a document using RAG.
    
    1. Embed the question
    2. Find the most relevant chunks
    3. Pass chunks + question to LLM
    """
    try:
        collection = chroma_client.get_collection(name=doc_id)
    except Exception:
        raise ValueError(f"Document '{doc_id}' not found. Please upload it first.")
    
    # Embed the question
    q_embedding = openai_client.embeddings.create(
        input=[question],
        model="text-embedding-3-small"
    ).data[0].embedding
    
    # Retrieve relevant chunks
    results = collection.query(
        query_embeddings=[q_embedding],
        n_results=min(n_results, collection.count())
    )
    
    retrieved_chunks = results["documents"][0] if results["documents"] else []
    context = "\n\n---\n\n".join(retrieved_chunks)
    
    # Generate answer using LLM
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an expert study assistant. Answer the student's question "
                    "based ONLY on the provided study material. If the answer isn't in "
                    "the material, say so honestly. Be clear, helpful, and educational.\n\n"
                    f"STUDY MATERIAL:\n{context}"
                )
            },
            {"role": "user", "content": question}
        ],
        temperature=0.4,
        max_tokens=800
    )
    
    return {
        "answer": response.choices[0].message.content,
        "sources": retrieved_chunks[:3],  # Show top sources
        "num_sources": len(retrieved_chunks)
    }


# =============================================
# 5. SUMMARIZE
# =============================================

def summarize_document(doc_id: str, openai_client) -> Dict:
    """Generate a comprehensive summary of the document."""
    try:
        collection = chroma_client.get_collection(name=doc_id)
    except Exception:
        raise ValueError(f"Document '{doc_id}' not found.")
    
    # Get all chunks
    all_data = collection.get()
    all_chunks = all_data["documents"] if all_data["documents"] else []
    
    # Combine text (limit to ~8000 words to fit context)
    combined = "\n\n".join(all_chunks)
    words = combined.split()
    if len(words) > 8000:
        combined = " ".join(words[:8000]) + "\n\n[...truncated for length]"
    
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a study assistant. Create a clear, well-structured summary "
                    "of the following study material. Use headings, bullet points, and "
                    "highlight key concepts. Make it useful for exam preparation."
                )
            },
            {
                "role": "user",
                "content": f"Please summarize this study material:\n\n{combined}"
            }
        ],
        temperature=0.3,
        max_tokens=1200
    )
    
    return {
        "summary": response.choices[0].message.content,
        "total_chunks": len(all_chunks),
        "words_processed": min(len(words), 8000)
    }


# =============================================
# 6. FLASHCARDS
# =============================================

def generate_flashcards(doc_id: str, openai_client, count: int = 10) -> Dict:
    """Generate flashcards from the document content."""
    try:
        collection = chroma_client.get_collection(name=doc_id)
    except Exception:
        raise ValueError(f"Document '{doc_id}' not found.")
    
    # Get all chunks
    all_data = collection.get()
    all_chunks = all_data["documents"] if all_data["documents"] else []
    
    combined = "\n\n".join(all_chunks)
    words = combined.split()
    if len(words) > 6000:
        combined = " ".join(words[:6000])
    
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a study assistant. Generate flashcards from the study material. "
                    "Each flashcard should have a 'front' (question/term) and 'back' (answer/definition). "
                    "Focus on the most important concepts, definitions, and key facts.\n\n"
                    "IMPORTANT: Return ONLY a valid JSON array, no markdown formatting.\n"
                    'Format: [{"front": "question", "back": "answer"}, ...]'
                )
            },
            {
                "role": "user",
                "content": f"Generate {count} flashcards from this material:\n\n{combined}"
            }
        ],
        temperature=0.5,
        max_tokens=1500
    )
    
    content = response.choices[0].message.content.strip()
    
    # Handle markdown code blocks
    if content.startswith("```"):
        lines = content.split("\n")
        lines = lines[1:]  # Remove first line
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        content = "\n".join(lines).strip()
    
    try:
        flashcards = json.loads(content)
        if not isinstance(flashcards, list):
            flashcards = []
    except json.JSONDecodeError:
        flashcards = []
    
    return {
        "flashcards": flashcards,
        "count": len(flashcards)
    }


# =============================================
# 7. DOCUMENT MANAGEMENT
# =============================================

def list_documents() -> List[Dict]:
    """List all uploaded documents."""
    metadata = _load_metadata()
    return list(metadata.values())


def delete_document(doc_id: str) -> bool:
    """Delete a document and its embeddings."""
    try:
        chroma_client.delete_collection(name=doc_id)
    except Exception:
        pass  # Collection might not exist
    
    metadata = _load_metadata()
    if doc_id in metadata:
        del metadata[doc_id]
        _save_metadata(metadata)
        return True
    return False


def get_document_info(doc_id: str) -> Optional[Dict]:
    """Get metadata for a specific document."""
    metadata = _load_metadata()
    return metadata.get(doc_id)
