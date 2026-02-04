import os
import numpy as np
import joblib
from keras.models import load_model

def get_binary_input(prompt):
    while True:
        value = input(prompt).strip()
        if value in ("0", "1"):
            return int(value)
        print("Invalid input. Please enter 0 (No) or 1 (Yes).")

def predict_covid(sample):
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    model = load_model(os.path.join(BASE_DIR, 'models', 'ann_covid_model.h5'))
    scaler = joblib.load(os.path.join(BASE_DIR, 'models', 'scaler.pkl'))

    sample = np.array(sample).reshape(1, -1)
    sample = scaler.transform(sample)

    prediction = model.predict(sample)[0][0]

    return "COVID-19 POSITIVE" if prediction >= 0.5 else "COVID-19 NEGATIVE"

if __name__ == "__main__":
    print("Enter symptoms (0 = No, 1 = Yes)\n")

    sample = [
        get_binary_input("Fever (0/1): "),
        get_binary_input("Cough (0/1): "),
        get_binary_input("Shortness of Breath (0/1): "),
        get_binary_input("Sore Throat (0/1): "),
        get_binary_input("Tiredness (0/1): "),
        get_binary_input("Body Pains (0/1): "),
        get_binary_input("Contact with COVID patient (0/1): "),
        get_binary_input("Gender Male (0/1): ")
    ]

    result = predict_covid(sample)
    print("\nPrediction:", result)