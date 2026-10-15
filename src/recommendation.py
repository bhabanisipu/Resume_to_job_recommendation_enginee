
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


def recommend_jobs(resume_text, resume_skills, jobs, tfidf, job_vectors, top_n=5):
    try:
        if not resume_text.strip():
            raise ValueError("Resume text cannot be empty.")

        if jobs.empty:
            raise ValueError("Jobs dataset is empty.")

        if len(jobs) != job_vectors.shape[0]:
            raise ValueError("Jobs and job vectors do not match.")

        resume_vector = tfidf.transform([resume_text.lower().strip()])

        similarity_scores = cosine_similarity(
            resume_vector, job_vectors
        ).flatten()

        resume_skills = {
            skill.strip().lower()
            for skill in resume_skills
            if skill.strip()
        }

        results = []

        for idx, (_, row) in enumerate(jobs.iterrows()):
            job_skills = {
                skill.strip().lower()
                for skill in str(row["skills"]).split("|")
                if skill.strip()
            }

            matched_skills = resume_skills.intersection(job_skills)

            skill_match_percentage = (
                len(matched_skills) / len(job_skills) * 100
                if job_skills else 0
            )

            combined_score = (
                0.5 * similarity_scores[idx]
                + 0.5 * (skill_match_percentage / 100)
            )

            results.append({
                "title": row["title"],
                "seniority": row["seniority"],
                "similarity_score": round(
                    float(similarity_scores[idx]), 4
                ),
                "matched_skills": ", ".join(sorted(matched_skills)),
                "skill_match_percentage": round(
                    skill_match_percentage, 2
                ),
                "combined_score": round(combined_score, 4)
            })

        recommendations = pd.DataFrame(results)

        return recommendations.sort_values(
            by="combined_score",
            ascending=False
        ).head(top_n).reset_index(drop=True)

    except Exception as e:
        print(f"Error recommending jobs: {e}")
        raise
