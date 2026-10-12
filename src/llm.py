"""One interface for many LLMs. Switch with LLM_PROVIDER in .env or per request.

Every provider is a plain REST call, so the only dependency is `requests`.
"""
import os

import requests

TIMEOUT = 120


def _openai_compatible(base_url, key_env, default_model):
    def call(system, user, history):
        key = os.getenv(key_env)
        if not key:
            raise RuntimeError(f"{key_env} is not set")
        model = os.getenv(f"{key_env.split('_')[0]}_MODEL", default_model)
        messages = [{"role": "system", "content": system}, *history,
                    {"role": "user", "content": user}]
        r = requests.post(f"{base_url}/chat/completions",
                          headers={"Authorization": f"Bearer {key}"},
                          json={"model": model, "messages": messages, "temperature": 0.2},
                          timeout=TIMEOUT)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]
    return call


def _anthropic(system, user, history):
    key = os.getenv("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set")
    r = requests.post("https://api.anthropic.com/v1/messages",
                      headers={"x-api-key": key, "anthropic-version": "2023-06-01"},
                      json={"model": os.getenv("ANTHROPIC_MODEL", "claude-haiku-5-5"),
                            "max_tokens": 1024, "temperature": 0.2, "system": system,
                            "messages": [*history, {"role": "user", "content": user}]},
                      timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()["content"][0]["text"]


def _gemini(system, user, history):
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("GEMINI_API_KEY is not set")
    model = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
    contents = [{"role": "model" if m["role"] == "assistant" else "user",
                 "parts": [{"text": m["content"]}]} for m in history]
    contents.append({"role": "user", "parts": [{"text": user}]})
    r = requests.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        headers={"x-goog-api-key": key},
        json={"system_instruction": {"parts": [{"text": system}]},
              "contents": contents, "generationConfig": {"temperature": 0.2}},
        timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()["candidates"][0]["content"]["parts"][0]["text"]


def _ollama(system, user, history):
    """Free, local, private. Install Ollama, then: ollama pull llama3.2"""
    host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    messages = [{"role": "system", "content": system}, *history,
                {"role": "user", "content": user}]
    r = requests.post(f"{host}/api/chat",
                      json={"model": os.getenv("OLLAMA_MODEL", "llama3.2"),
                            "messages": messages, "stream": False,
                            "options": {"temperature": 0.2}},
                      timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()["message"]["content"]


def _offline(system, user, history):
    """No LLM at all: returns the retrieved passages. Handy for testing retrieval."""
    context = user.split("CONTEXT:", 1)[-1].split("QUESTION:", 1)[0].strip()
    return "(offline mode - no LLM, showing retrieved passages)\n\n" + context


PROVIDERS = {
    "openai": _openai_compatible("https://api.openai.com/v1", "OPENAI_API_KEY", "gpt-4o-mini"),
    "groq": _openai_compatible("https://api.groq.com/openai/v1", "GROQ_API_KEY",
                               "llama-3.3-70b-versatile"),
    "anthropic": _anthropic,
    "gemini": _gemini,
    "ollama": _ollama,
    "offline": _offline,
}


def generate(system, user, history=None, provider=None):
    provider = (provider or os.getenv("LLM_PROVIDER", "offline")).lower()
    if provider not in PROVIDERS:
        raise ValueError(f"Unknown provider '{provider}'. Choose from {list(PROVIDERS)}")
    return PROVIDERS[provider](system, user, history or [])
