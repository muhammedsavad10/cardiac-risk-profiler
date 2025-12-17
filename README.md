# 🫀 Cardiac Risk Profiler

A Machine Learning application that predicts the likelihood of Coronary Artery Disease (CAD) based on clinical parameters.

## 🚀 Overview
Unlike typical "black box" AI models, this project prioritizes **medical interpretability**. It uses **Logistic Regression** to not only predict risk but also explain *which* factors (e.g., Chest Pain Type, Thalassemia) contribute most to the diagnosis.

## 🛠️ Tech Stack
* **Python 3.9+**
* **Streamlit** (Frontend Interface)
* **Scikit-Learn** (Machine Learning)
* **Pandas & NumPy** (Data Processing)

## 📊 How It Works
1.  **Data Ingestion:** Loads the UCI Cleveland Heart Disease dataset.
2.  **Preprocessing:** Scales features using `StandardScaler` to ensure fair weighting.
3.  **Modeling:** Trains a Logistic Regression model (Accuracy ~85%).
4.  **Inference:** Accepts user input via the dashboard and outputs a probabilistic risk score.

## 💻 How to Run Locally
1. Clone the repository:
   git clone [https://github.com/muhammedsavad10/cardiac-risk-profiler.git](https://github.com/muhammedsavad10/cardiac-risk-profiler.git)

2.Install dependencies:
pip install -r requirements.txt

3.Run the app:
streamlit run app.py

