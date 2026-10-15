
from src.data_preprocessing import load_job_data, preprocess_job_data
from src.feature_engineering import create_job_vectors
from src.recommendation import recommend_jobs


def main():
    try:
        df = load_job_data()
        jobs = preprocess_job_data(df)

        tfidf, job_vectors = create_job_vectors(jobs)

        resume_text = input("Enter your resume text: ").strip()

        skills_input = input(
            "Enter your skills separated by commas: "
        ).strip()

        resume_skills = [
            skill.strip().lower()
            for skill in skills_input.split(",")
            if skill.strip()
        ]

        # Step 4: Recommend jobs
        recommendations = recommend_jobs(
            resume_text=resume_text,
            resume_skills=resume_skills,
            jobs=jobs,
            tfidf=tfidf,
            job_vectors=job_vectors,
            top_n=5
        )

        # Step 5: Display recommendations
        print("\nTop 5 Recommended Jobs:\n")
        print(recommendations.to_string(index=False))

    except Exception as e:
        print(f"Application failed: {e}")


if __name__ == "__main__":
    main()
