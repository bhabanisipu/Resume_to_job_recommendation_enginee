# Resume-to-Job Recommendation Engine

## 📌 Overview
A Machine Learning-based web application that recommends relevant job opportunities based on a candidate's resume and technical skills. It uses NLP and similarity-based matching to rank jobs and display the top 5 recommendations.

## 🚀 Features
- Resume-based job recommendations
- TF-IDF text vectorization
- Cosine Similarity for job matching
- Technical skill matching and percentage calculation
- Combined scoring and job ranking
- Interactive Flask web interface

## 🛠️ Technologies Used
- **Language:** Python
- **ML/NLP:** Scikit-learn, TF-IDF, Cosine Similarity
- **Data Processing:** Pandas
- **Backend:** Flask
- **Frontend:** HTML, CSS, Bootstrap
- **Model Persistence:** Joblib

## 📂 Project Structure
```text
resume_job_recommendation/
├── data/
├── models/
├── notebooks/
├── src/
├── templates/
├── static/
├── app.py
└── requirements.txt
```

## ⚙️ Installation

1. Clone the repository:
   ```bash
   git clone <your-repository-url>
   ```

2. Navigate to the project folder:
   ```bash
   cd resume_job_recommendation
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the application:
   ```bash
   python app.py
   ```

5. Open `http://127.0.0.1:5000/` in your browser.

## 💡 How It Works
1. Enter your resume text and technical skills.
2. The system processes the text using TF-IDF.
3. Cosine Similarity measures resume-to-job text similarity.
4. Skill matching calculates the overlap between candidate skills and job requirements.
5. The system ranks and displays the top 5 job recommendations.

## 🎯 Objective
To simplify job discovery by matching candidates with relevant opportunities using Machine Learning and NLP.

**Note:** Recommendation scores represent relative matching strength, not the probability of getting hired.