
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "data-job-postings.csv"


def load_job_data():
    try:
        df = pd.read_csv(DATA_PATH)

        if df.empty:
            raise ValueError("Dataset is empty.")

        print("Dataset loaded successfully!")
        return df

    except FileNotFoundError as e:
        print(f"Dataset file not found: {e}")
        raise

    except pd.errors.EmptyDataError as e:
        print(f"CSV file is empty: {e}")
        raise

    except pd.errors.ParserError as e:
        print(f"CSV parsing error: {e}")
        raise

    except Exception as e:
        print(f"Error loading dataset: {e}")
        raise


def preprocess_job_data(df):
    try:
        required_columns = ["title", "skills", "seniority"]

        missing_columns = [
            col for col in required_columns
            if col not in df.columns
        ]

        if missing_columns:
            raise ValueError(
                f"Missing required columns: {missing_columns}"
            )

        jobs = df[required_columns].copy()

        for col in required_columns:
            jobs[col] = (
                jobs[col]
                .fillna("")
                .astype(str)
                .str.lower()
                .str.strip()
            )

        jobs["skills_text"] = (
            jobs["skills"]
            .str.replace("|", " ", regex=False)
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
        )

        jobs["job_text"] = (
            jobs["title"] + " " + jobs["skills_text"]
        ).str.strip()

        print("Preprocessing completed successfully!")
        return jobs

    except Exception as e:
        print(f"Error during preprocessing: {e}")
        raise


if __name__ == "__main__":
    try:
        df = load_job_data()
        jobs = preprocess_job_data(df)

        print("\nDataset shape:", jobs.shape)
        print("\nColumns:", jobs.columns.tolist())
        print("\nSample data:")
        print(jobs.head())

    except Exception as e:
        print(f"Execution failed: {e}")
