import os
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request
from werkzeug.utils import secure_filename

load_dotenv()

from src import testlab  # noqa: E402
from src.llm import PROVIDERS  # noqa: E402
from src.pipeline import Assistant  # noqa: E402
from src.rag import DATA_DIR, build_index  # noqa: E402

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024
assistant = Assistant()


@app.get("/")
def index():
    return render_template("index.html", providers=list(PROVIDERS),
                           default=os.getenv("LLM_PROVIDER", "offline"))


@app.post("/chat")
def chat():
    data = request.get_json(force=True)
    message = (data.get("message") or "").strip()
    if not message:
        return jsonify(error="Empty message"), 400
    try:
        return jsonify(assistant.ask(message, data.get("history", [])[-6:], data.get("provider")))
    except Exception as e:  # missing key, Ollama not running, no index, ...
        return jsonify(error=f"{type(e).__name__}: {e}"), 500


@app.get("/api/status")
def status():
    return jsonify(providers=testlab.provider_status(),
                   index_chunks=len(assistant.retriever.chunks) if assistant.retriever else 0,
                   ml_model=bool(assistant.predictor),
                   documents=sorted(p.name for p in DATA_DIR.glob("*") if p.suffix.lower() in {".pdf", ".txt", ".md"}))


@app.post("/api/keys")
def keys():
    testlab.set_keys(request.get_json(force=True))
    return jsonify(testlab.provider_status())


@app.post("/api/compare")
def compare():
    data = request.get_json(force=True)
    try:
        return jsonify(testlab.compare(assistant, data["message"], data["providers"]))
    except Exception as e:
        return jsonify(error=f"{type(e).__name__}: {e}"), 500


@app.post("/api/upload")
def upload():
    DATA_DIR.mkdir(exist_ok=True)
    saved = []
    for f in request.files.getlist("files"):
        name = secure_filename(f.filename or "")
        if Path(name).suffix.lower() in {".pdf", ".txt", ".md"}:
            f.save(DATA_DIR / name)
            saved.append(name)
    if not saved:
        return jsonify(error="Upload .pdf, .txt or .md files"), 400
    chunks = build_index()
    assistant.reload()
    return jsonify(saved=saved, chunks=chunks)


@app.post("/api/reindex")
def reindex():
    chunks = build_index()
    assistant.reload()
    return jsonify(chunks=chunks)


@app.get("/api/repo-check")
def repo_check():
    try:
        return jsonify(testlab.repo_audit(request.args.get("repo", "")))
    except Exception as e:
        return jsonify(error=f"{type(e).__name__}: {e}"), 500


@app.get("/api/eval")
def run_eval():
    return jsonify(testlab.retrieval_eval(assistant))


if __name__ == "__main__":
    app.run(debug=False, port=8080)
