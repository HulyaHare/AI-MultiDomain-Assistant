import numpy as np
from app.data.loader import DataStore


def predict_disease(symptoms_text: str) -> dict:
    """
    Accept free-text symptoms, vectorize with the fitted TF-IDF,
    predict disease via LinearSVC, and return structured result.
    """
    vec = DataStore.health_vectorizer.transform([symptoms_text])
    pred_encoded = DataStore.health_model.predict(vec)[0]
    disease = DataStore.health_label_encoder.inverse_transform([pred_encoded])[0]

    decision = DataStore.health_model.decision_function(vec)
    if decision.ndim == 1:
        confidence = float(np.max(decision))
    else:
        confidence = float(np.max(decision[0]))

    disease_cases = DataStore.health_df[DataStore.health_df["Disease"] == disease]

    all_symptoms = disease_cases["Symptoms"].dropna().tolist()
    from collections import Counter
    sym_counter = Counter()
    for s in all_symptoms:
        for token in s.split(","):
            token = token.strip().lower()
            if token:
                sym_counter[token] += 1
    common_symptoms = [s for s, _ in sym_counter.most_common(10)]

    return {
        "predicted_disease": disease,
        "confidence": round(confidence, 3),
        "common_symptoms": common_symptoms,
        "total_cases_in_data": len(disease_cases),
    }
