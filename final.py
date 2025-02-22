import pandas as pd
import os
import joblib
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import GridSearchCV

# Load dataset
def load_dataset():
    df = pd.read_csv("diseases_symptoms_35_utf8.csv")
    df.columns = df.columns.str.strip()
    df["Symptoms"] = df["Symptoms"].apply(lambda x: x.split(", "))
    return df

# Train model
def train_model(df):
    mlb = MultiLabelBinarizer()
    X = mlb.fit_transform(df["Symptoms"])
    y = df["Disease"]

    print("Class distribution:\n", y.value_counts())

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(random_state=42)
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }

    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=kf,
        n_jobs=-1,
        verbose=2
    )
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_

    y_pred = best_model.predict(X_test)
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")

    joblib.dump((best_model, mlb), "disease_predictor.pkl")
    print("Model trained and saved as disease_predictor.pkl")

# Predict disease using ML model
def predict_disease(user_symptoms, mlb, model):
    all_symptoms = mlb.classes_
    user_input_vector = np.zeros(len(all_symptoms))
    for symptom in user_symptoms:
        if symptom in all_symptoms:
            index = np.where(all_symptoms == symptom)[0][0]
            user_input_vector[index] = 1
        else:
            st.error(f"Symptom '{symptom}' not recognized.")
    user_input_vector = user_input_vector.reshape(1, -1)
    probabilities = model.predict_proba(user_input_vector)
    top_indices = np.argsort(probabilities[0])[::-1][:3]
    top_diseases = model.classes_[top_indices]
    confidences = probabilities[0][top_indices]
    predictions = list(zip(top_diseases, confidences))
    return predictions

# Helper function to ask and validate a response for a symptom
def get_response(symptom, iteration):
    while True:
        try:
            response = int(st.text_input(
                f"Rate {symptom}: (0 = No, 1 = Mild, 2 = Moderate, 3 = Severe)",
                key=f"{symptom}_{iteration}"  # Unique key based on symptom and iteration
            ))
            if response in [0, 1, 2, 3]:
                return response
            else:
                st.error("Please enter a valid number (0-3).")
        except ValueError:
            st.error("Invalid input. Please enter a number between 0 and 3.")

# Targeted adaptive questioning that focuses on missing symptoms of the top candidate diseases
def ask_questions_targeted_union(df, max_iterations=10):
    symptoms_map = {row["Disease"]: set(row["Symptoms"]) for _, row in df.iterrows()}
    all_symptoms = {symptom for symptoms in symptoms_map.values() for symptom in symptoms}

    asked_symptoms = {}
    st.write("\nPlease answer the following questions on a scale of 0 to 3 (0 = No, 1 = Mild, 2 = Moderate, 3 = Severe)")

    general_symptoms = ["Fever", "Fatigue", "Cough", "Headache", "Nausea"]
    for i, symptom in enumerate(general_symptoms):
        if symptom in all_symptoms:
            asked_symptoms[symptom] = get_response(symptom, i)

    positive_symptoms = {s for s, resp in asked_symptoms.items() if resp > 0}
    if positive_symptoms:
        candidate_diseases = {d for d, s_set in symptoms_map.items() if any(s in s_set for s in positive_symptoms)}
    else:
        candidate_diseases = set(symptoms_map.keys())

    iteration = len(general_symptoms)
    while iteration < max_iterations and len(candidate_diseases) > 1:
        def match_score(d):
            score = 0
            for s in symptoms_map[d]:
                if s in asked_symptoms and asked_symptoms[s] > 0:
                    score += asked_symptoms[s]
            return score
        candidate_scores = {d: match_score(d) for d in candidate_diseases}
        max_score = max(candidate_scores.values())
        top_candidates = {d for d, score in candidate_scores.items() if score == max_score}

        missing_symptoms = set()
        for d in top_candidates:
            missing_symptoms |= (symptoms_map[d] - set(asked_symptoms.keys()))
        if not missing_symptoms:
            break

        for symptom in missing_symptoms:
            response = get_response(symptom, iteration)
            asked_symptoms[symptom] = response
            if response > 0:
                candidate_diseases = {d for d in candidate_diseases if symptom in symptoms_map[d]}
        iteration += 1

    return asked_symptoms, candidate_diseases

def main():
    st.title("Disease Prediction Based on Symptoms")

    # Load dataset and model
    df = load_dataset()

    if not os.path.exists("disease_predictor.pkl"):
        train_model(df)
    else:
        st.write("Model already trained. Skipping training.")

    model, mlb = joblib.load("disease_predictor.pkl")

    # Use the targeted adaptive questioning function
    asked_symptoms, candidate_diseases = ask_questions_targeted_union(df, max_iterations=10)
    
    # Use only the positive symptoms for ML model prediction.
    user_symptoms = [symptom for symptom, resp in asked_symptoms.items() if resp > 0]
    if user_symptoms:
        predictions = predict_disease(user_symptoms, mlb, model)
        st.write("\n**ML Model Predictions:**")
        for disease, confidence in predictions:
            st.write(f"- {disease} (Confidence: {confidence:.2f})")
    else:
        st.write("\nNo positive symptoms detected for ML prediction.")

if __name__ == "__main__":
    main()
