"""User message -> (optional ML prediction) -> retrieval -> LLM -> answer + sources."""
from . import llm
from .predictor import MODEL_FILE, DiseasePredictor
from .rag import INDEX_FILE, Retriever

DISCLAIMER = ("This is educational information, not a medical diagnosis. "
              "Please consult a qualified doctor. In an emergency call your local emergency number.")

SYSTEM_PROMPT = """You are a careful medical information assistant for an educational project.
Rules:
- Answer ONLY from the CONTEXT passages. If the context does not contain the answer, say you
  don't have enough trusted information and suggest seeing a doctor.
- Never give a confirmed diagnosis. Say "possible" / "may be related to".
- Cite sources as [1], [2] matching the numbered passages.
- Structure: Possible conditions, Common symptoms, General precautions, Next steps,
  When to see a doctor urgently.
- Use simple, plain language. Do not recommend specific drug doses."""

NO_CONTEXT = ("I couldn't find relevant information in my trusted medical documents. "
              "Please consult a doctor.")


class Assistant:
    def __init__(self):
        self.reload()

    def reload(self):
        """(Re)load index + ML model from disk. Missing index is allowed: the app
        still starts and the UI tells the user to add documents."""
        self.retriever = Retriever() if INDEX_FILE.exists() else None
        self.predictor = DiseasePredictor() if MODEL_FILE.exists() else None

    def prepare(self, message):
        """Everything before the LLM call. Returns (prompt, hits, predictions)."""
        if not self.retriever:
            raise RuntimeError("No documents indexed yet. Upload documents in the Documents tab.")
        predictions, query = [], message
        if self.predictor:
            predictions = self.predictor.predict(self.predictor.extract_symptoms(message))
            if predictions:  # enrich retrieval with the ML output
                query += " " + " ".join(d for d, _ in predictions)
        hits = self.retriever.search(query)
        if not hits:
            return None, hits, predictions
        context = "\n\n".join(f"[{i}] ({h['source']} {h['page']}) {h['text']}"
                              for i, h in enumerate(hits, 1))
        ml_note = ""
        if predictions:
            ml_note = "ML model's possible diseases (not a diagnosis): " + ", ".join(
                f"{d} ({p:.0%})" for d, p in predictions) + "\n\n"
        return f"{ml_note}CONTEXT:\n{context}\n\nQUESTION: {message}", hits, predictions

    @staticmethod
    def package(answer, hits, predictions):
        return {"answer": answer,
                "sources": [{"id": i, "source": h["source"], "page": h["page"],
                             "score": round(h["score"], 3)} for i, h in enumerate(hits, 1)],
                "predictions": [{"disease": d, "probability": round(p, 3)} for d, p in predictions],
                "disclaimer": DISCLAIMER}

    def ask(self, message, history=None, provider=None):
        prompt, hits, predictions = self.prepare(message)
        if prompt is None:
            return self.package(NO_CONTEXT, [], predictions)
        answer = llm.generate(SYSTEM_PROMPT, prompt, history, provider)
        return self.package(answer, hits, predictions)
