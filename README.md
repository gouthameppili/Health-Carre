﻿# Health-Carre
# 🩺 Health-Carre – Adaptive Disease Prediction App

Health-Carre is a Streamlit-based web application that predicts diseases based on user-reported symptoms using a machine learning model. It uses **Random Forest**, **multi-label symptom encoding**, and **adaptive questioning** to improve diagnosis accuracy and user experience.

---

## 🚀 Features

- ✅ Adaptive symptom questioning (dynamic based on your answers)
- ✅ Machine learning–based disease prediction
- ✅ Top 3 disease predictions with confidence scores
- ✅ Symptom severity input (0–3 scale)
- ✅ Built with Streamlit, Scikit-learn, Pandas, and Joblib

---

## 📊 Dataset

The application uses a CSV dataset:  
`diseases_symptoms_35_utf8.csv`  
- Contains 35 diseases and their associated symptoms.
- Symptoms are stored as comma-separated lists.
- The model uses multi-label binarization to encode symptom presence.

---

## 🛠️ Tech Stack

| Component        | Description                        |
|------------------|------------------------------------|
| 🧠 ML Model       | Random Forest Classifier            |
| 📊 Preprocessing | MultiLabelBinarizer + GridSearchCV  |
| 📈 Metrics       | Accuracy + Classification Report    |
| 💻 Frontend      | Streamlit UI                        |
| 💾 Storage       | `joblib` to save trained model      |

---

## 🖼️ Demo Screenshot

*(Insert a screenshot here if available)*

---

## 🧪 How It Works

1. Loads the dataset (`diseases_symptoms_35_utf8.csv`)
2. Trains or loads a `RandomForestClassifier` stored in `disease_predictor.pkl`
3. Asks general symptoms first (Fever, Cough, etc.)
4. Based on responses, narrows down the possible diseases
5. Asks additional targeted questions to further filter
6. Predicts top 3 diseases using trained model and shows confidence

---

## 💻 Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/gouthameppili/Health-Carre.git
cd Health-Carre
