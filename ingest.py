"""Build the vector index from everything in data/.  Usage: python ingest.py"""
from dotenv import load_dotenv

load_dotenv()
from src.rag import build_index  # noqa: E402

try:
    print(f"Indexed {build_index()} chunks.")
except ValueError as e:
    raise SystemExit(str(e))
