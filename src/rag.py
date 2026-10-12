"""Ingest trusted documents -> chunks -> vector index -> retrieval.

Default embedding is TF-IDF (no model download, works anywhere). For better
semantic search set EMBEDDER=st (needs `pip install sentence-transformers`).
"""
import os
import pickle
import re
from pathlib import Path

import numpy as np
from pypdf import PdfReader

DATA_DIR = Path("data")
INDEX_FILE = Path("index/index.pkl")
CHUNK_WORDS, OVERLAP = 180, 40


def load_documents(data_dir=DATA_DIR):
    """Yield (source_name, page_label, text) for every .pdf / .txt / .md file."""
    for path in sorted(Path(data_dir).glob("**/*")):
        if path.suffix.lower() == ".pdf":
            for i, page in enumerate(PdfReader(path).pages, 1):
                yield path.name, f"p.{i}", page.extract_text() or ""
        elif path.suffix.lower() in {".txt", ".md"}:
            yield path.name, "", path.read_text(encoding="utf-8", errors="ignore")


def chunk(text, size=CHUNK_WORDS, overlap=OVERLAP):
    """One chunk per paragraph; paragraphs longer than `size` words are windowed."""
    step = size - overlap
    for para in re.split(r"\n\s*\n", text):
        words = para.split()
        for start in range(0, len(words), step):
            piece = " ".join(words[start:start + size])
            if len(piece.split()) >= 15:
                yield piece
            if start + size >= len(words):
                break


class Embedder:
    def __init__(self, kind=None):
        self.kind = kind or os.getenv("EMBEDDER", "tfidf")
        self.model = None

    def fit_transform(self, texts):
        if self.kind == "st":
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            return self.model.encode(texts, normalize_embeddings=True)
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.model = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
        return self.model.fit_transform(texts)  # sparse, L2-normalised

    def transform(self, texts):
        if self.kind == "st":
            return self.model.encode(texts, normalize_embeddings=True)
        return self.model.transform(texts)


def build_index():
    chunks, meta = [], []
    for source, page, text in load_documents():
        for piece in chunk(text):
            chunks.append(piece)
            meta.append({"source": source, "page": page})
    if not chunks:
        raise ValueError(f"No documents found in {DATA_DIR}/. Add PDF/TXT files first.")
    embedder = Embedder()
    matrix = embedder.fit_transform(chunks)
    INDEX_FILE.parent.mkdir(exist_ok=True)
    with open(INDEX_FILE, "wb") as f:
        pickle.dump({"embedder": embedder, "matrix": matrix, "chunks": chunks, "meta": meta}, f)
    return len(chunks)


class Retriever:
    def __init__(self):
        if not INDEX_FILE.exists():
            raise FileNotFoundError("No index yet. Run: python ingest.py")
        with open(INDEX_FILE, "rb") as f:
            idx = pickle.load(f)
        self.embedder, self.matrix = idx["embedder"], idx["matrix"]
        self.chunks, self.meta = idx["chunks"], idx["meta"]

    def search(self, query, k=4, min_score=0.05):
        q = self.embedder.transform([query])
        scores = (self.matrix @ q.T)
        scores = scores.toarray().ravel() if hasattr(scores, "toarray") else np.ravel(scores)
        top = np.argsort(scores)[::-1][:k]
        return [{"text": self.chunks[i], "score": float(scores[i]), **self.meta[i]}
                for i in top if scores[i] >= min_score]
