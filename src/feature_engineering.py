
from pathlib import Path
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"
VECTORIZER_PATH = MODEL_DIR / "tfidf_vectorizer.pkl"


def create_job_vectors(jobs):
    try:
        if "job_text" not in jobs.columns:
            raise ValueError("The 'job_text' column is missing.")

        if jobs.empty:
            raise ValueError("The jobs dataset is empty.")

        tfidf = TfidfVectorizer()
        job_vectors = tfidf.fit_transform(
            jobs["job_text"].fillna("")
        )

        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(tfidf, VECTORIZER_PATH)

        print("TF-IDF vectorizer trained successfully!")
        print("Job vectors shape:", job_vectors.shape)

        return tfidf, job_vectors

    except Exception as e:
        print(f"Error creating job vectors: {e}")
        raise
