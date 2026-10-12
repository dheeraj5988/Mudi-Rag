"""Helpers behind the Test Lab tab: provider status, side-by-side compare, repo audit, retrieval eval."""
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

from . import llm
from .pipeline import SYSTEM_PROMPT

KEY_ENV = {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY",
           "gemini": "GEMINI_API_KEY", "groq": "GROQ_API_KEY"}
EVAL_FILE = Path("data/eval.json")


def ollama_models():
    host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    try:
        r = requests.get(f"{host}/api/tags", timeout=2)
        return [m["name"] for m in r.json().get("models", [])]
    except Exception:
        return None


def provider_status():
    out = {"offline": {"ready": True, "note": "always available (shows retrieved text only)"}}
    for name, env in KEY_ENV.items():
        ok = bool(os.getenv(env))
        out[name] = {"ready": ok, "note": "key set" if ok else f"{env} not set"}
    models = ollama_models()
    want = os.getenv("OLLAMA_MODEL", "llama3.2")
    if models is None:
        out["ollama"] = {"ready": False, "note": "Ollama not running on localhost:11434"}
    elif not any(m.startswith(want) for m in models):
        out["ollama"] = {"ready": False, "note": f"running, but model '{want}' not pulled "
                                                 f"(have: {', '.join(models) or 'none'})"}
    else:
        out["ollama"] = {"ready": True, "note": f"running, model '{want}' available"}
    return out


def set_keys(keys):
    """Keep keys in this process's memory only; never written to disk or logged."""
    for name, value in keys.items():
        env = KEY_ENV.get(name)
        if env and value.strip():
            os.environ[env] = value.strip()


def groundedness(answer, hits):
    """Cheap heuristic (NOT a real faithfulness metric): share of the answer's longer words
    that also appear in the retrieved passages, plus whether it cites [n]."""
    words = {w for w in re.findall(r"[a-z]{5,}", answer.lower())}
    ctx = " ".join(h["text"] for h in hits).lower()
    ctx_words = set(re.findall(r"[a-z]{5,}", ctx))
    overlap = len(words & ctx_words) / len(words) if words else 0.0
    return {"overlap": round(overlap, 2), "cites": bool(re.search(r"\[\d+\]", answer))}


def compare(assistant, message, providers):
    prompt, hits, _ = assistant.prepare(message)
    if prompt is None:
        return {"error": "No relevant passages found - nothing to send to the LLMs.", "results": []}

    def run(p):
        t0 = time.time()
        try:
            ans = llm.generate(SYSTEM_PROMPT, prompt, None, p)
            return {"provider": p, "ok": True, "answer": ans,
                    "seconds": round(time.time() - t0, 2), **groundedness(ans, hits)}
        except Exception as e:
            return {"provider": p, "ok": False, "seconds": round(time.time() - t0, 2),
                    "error": f"{type(e).__name__}: {str(e)[:300]}"}

    with ThreadPoolExecutor(max_workers=max(len(providers), 1)) as pool:
        results = list(pool.map(run, providers))
    return {"results": results,
            "sources": [{"id": i, "source": h["source"], "page": h["page"]} for i, h in enumerate(hits, 1)]}


def repo_audit(repo):
    """Inspect a public GitHub repo via the GitHub API: which files exist, which are empty."""
    if not re.fullmatch(r"[\w.-]+/[\w.-]+", repo):
        return {"error": "Use the form owner/name"}
    r = requests.get(f"https://api.github.com/repos/{repo}/git/trees/HEAD?recursive=1", timeout=15)
    if r.status_code != 200:
        return {"error": f"GitHub returned {r.status_code} (private repo or wrong name?)"}
    files = [t for t in r.json()["tree"] if t["type"] == "blob"]
    code = [f for f in files if f["path"].endswith((".py", ".ipynb", ".sh", ".html", ".js"))]
    empty = [f["path"] for f in code if f["size"] == 0]
    names = {f["path"] for f in files}
    findings = []
    if empty:
        findings.append(f"{len(empty)} of {len(code)} code files are EMPTY: {', '.join(empty)}")
    for f in files:
        if f["path"].endswith("_init_.py"):
            findings.append(f"'{f['path']}' has single underscores; Python needs __init__.py for a package")
    if "app.py" in names and not any(f["path"] == "app.py" and f["size"] > 0 for f in files):
        findings.append("app.py is empty: there is no web app to run")
    if not any(p.startswith("data/") for p in names):
        findings.append("no data/ folder: there are no medical documents to retrieve from")
    req = requests.get(f"https://raw.githubusercontent.com/{repo}/HEAD/requirements.txt", timeout=15)
    if req.ok and "pinecone-client" in req.text:
        findings.append("requirements.txt uses 'pinecone-client' (deprecated; the package is now 'pinecone')")
    verdict = "NOT RUNNABLE" if (empty and ("app.py" in empty)) else "needs manual check"
    return {"repo": repo, "verdict": verdict, "findings": findings,
            "files": [{"path": f["path"], "bytes": f["size"]} for f in files]}


def retrieval_eval(assistant):
    """data/eval.json: [{"question": "...", "expect": ["keyword", ...]}]. A hit = the single best
    passage (top-1) contains every expected keyword."""
    if not assistant.retriever:
        return {"error": "No index yet."}
    if not EVAL_FILE.exists():
        return {"error": f"{EVAL_FILE} not found"}
    cases = json.loads(EVAL_FILE.read_text())
    rows, hits_n = [], 0
    for c in cases:
        found = assistant.retriever.search(c["question"], k=1)
        text = " ".join(h["text"] for h in found).lower()
        hit = all(k.lower() in text for k in c["expect"])
        hits_n += hit
        rows.append({"question": c["question"], "expect": c["expect"], "hit": hit,
                     "top_source": found[0]["source"] if found else None})
    return {"hit_rate": round(hits_n / len(cases), 2) if cases else 0, "cases": rows}
