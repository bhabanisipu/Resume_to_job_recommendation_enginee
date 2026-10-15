
from flask import Flask, render_template, request
from src.data_preprocessing import load_job_data, preprocess_job_data
from src.feature_engineering import create_job_vectors
from src.recommendation import recommend_jobs

app = Flask(__name__, template_folder="template", static_folder="static")

try:
    # Load and preprocess job data
    df = load_job_data()
    jobs = preprocess_job_data(df)

    # Create TF-IDF vectors
    tfidf, job_vectors = create_job_vectors(jobs)

except Exception as e:
    raise RuntimeError(f"Failed to initialize the application: {e}")


@app.route("/", methods=["GET", "POST"])
def home():
    recommendations = None
    error = None

    if request.method == "POST":
        try:
            resume_text = request.form.get("resume_text", "").strip()
            skills_input = request.form.get("skills", "").strip()

            if not resume_text:
                raise ValueError("Please enter your resume text.")

            if not skills_input:
                raise ValueError("Please enter your skills.")

            resume_skills = [
                skill.strip().lower()
                for skill in skills_input.split(",")
                if skill.strip()
            ]

            recommendations = recommend_jobs(
                resume_text=resume_text,
                resume_skills=resume_skills,
                jobs=jobs,
                tfidf=tfidf,
                job_vectors=job_vectors,
                top_n=5
            )

            recommendations = recommendations.to_dict(orient="records")

        except ValueError as e:
            error = str(e)

        except Exception as e:
            app.logger.exception("Recommendation failed")
            error = "Something went wrong while generating recommendations."

    return render_template(
        "index.html",
        recommendations=recommendations,
        error=error
    )


if __name__ == "__main__":
    app.run(debug=True)
