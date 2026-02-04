import os
import joblib
from data_preprocessing import load_and_preprocess_data
from ann_model import build_ann

def train_model():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    dataset_path = os.path.join(BASE_DIR, 'dataset', 'covid_data.csv')
    model_dir = os.path.join(BASE_DIR, 'models')

    os.makedirs(model_dir, exist_ok=True)

    X_train, X_test, y_train, y_test, scaler = load_and_preprocess_data(dataset_path)

    model = build_ann(X_train.shape[1])

    model.fit(
        X_train,
        y_train,
        epochs=15,
        batch_size=32,
        validation_split=0.1,
        verbose=1
    )

    model.save(os.path.join(model_dir, 'ann_covid_model.h5'))
    joblib.dump(scaler, os.path.join(model_dir, 'scaler.pkl'))

    print("Model and scaler saved successfully.")

if __name__ == "__main__":
    train_model()
