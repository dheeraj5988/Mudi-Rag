"""Optional ML step: Random Forest over symptom flags -> top-k likely diseases.

Expected training CSV (e.g. Kaggle "Disease Prediction Using Machine Learning"
Training.csv): one 0/1 column per symptom plus a `prognosis` label column.
"""
import re
import pickle
from pathlib import Path

MODEL_FILE = Path("index/disease_model.pkl")


def train(csv_path, label="prognosis"):
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report
    from sklearn.model_selection import train_test_split

    df = pd.read_csv(csv_path).dropna(axis=1, how="all")
    X, y = df.drop(columns=[label]), df[label]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    clf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1).fit(X_tr, y_tr)
    print(classification_report(y_te, clf.predict(X_te), zero_division=0))
    MODEL_FILE.parent.mkdir(exist_ok=True)
    with open(MODEL_FILE, "wb") as f:
        pickle.dump({"clf": clf, "symptoms": list(X.columns)}, f)


def _norm(s):
    return re.sub(r"[^a-z]+", " ", s.lower()).strip()


class DiseasePredictor:
    def __init__(self):
        with open(MODEL_FILE, "rb") as f:
            m = pickle.load(f)
        self.clf, self.symptoms = m["clf"], m["symptoms"]
        self._lookup = {_norm(s): s for s in self.symptoms}

    def extract_symptoms(self, text):
        """Keyword match of free text against the dataset's symptom names.
        (Swap for an LLM extractor if you want fuzzier matching.)"""
        t = " " + _norm(text) + " "
        return [orig for key, orig in self._lookup.items() if f" {key} " in t]

    def predict(self, symptoms, k=3):
        if not symptoms:
            return []
        import pandas as pd
        row = pd.DataFrame([[1 if s in symptoms else 0 for s in self.symptoms]],
                           columns=self.symptoms)
        proba = self.clf.predict_proba(row)[0]
        top = proba.argsort()[::-1][:k]
        return [(self.clf.classes_[i], float(proba[i])) for i in top if proba[i] > 0]
