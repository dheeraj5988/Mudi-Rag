# Medi-RAG: medical chatbot with RAG, a switchable LLM and optional ML disease prediction

Educational prototype. It does not diagnose.

```
User message -> (optional) symptom extraction + Random Forest -> retrieve from trusted docs
             -> LLM (openai | anthropic | gemini | groq | ollama | offline) -> answer + sources + disclaimer
```

## Run
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env            # set LLM_PROVIDER and the matching key
# put trusted PDFs/TXT (WHO, CDC, NHS, MedlinePlus) in data/ and delete SAMPLE_replace_me.txt
python ingest.py                # builds the index
python train_model.py Training.csv   # OPTIONAL: Kaggle symptom CSV with a 'prognosis' column
python app.py                   # http://localhost:8080 (Test Lab)
```

## Test Lab (http://localhost:8080)
Tabs: **Chat** | **Compare LLMs** (same question + same retrieved context to every provider; shows which work, latency, citations)
| **Documents & status** (upload PDFs, see what is ready) | **API keys** (kept in memory only) |
**GitHub repo check** (flags empty/broken files in any public repo) | **Retrieval test** (`data/eval.json`, top-1 hit rate).

## Switching LLMs
Set `LLM_PROVIDER` in `.env`, or use the dropdown in the UI. `ollama` is free and local
(`ollama pull llama3.2`). `offline` needs nothing and just shows retrieved passages, which is good for debugging retrieval.
To add another LLM, add one function to `src/llm.py` and register it in `PROVIDERS`.

## Files
- `src/rag.py` load, chunk, embed, retrieve (TF-IDF by default, `EMBEDDER=st` for sentence-transformers)
- `src/predictor.py` Random Forest symptom classifier
- `src/pipeline.py` glues it together; the grounding prompt is here
- `src/llm.py` provider adapters
