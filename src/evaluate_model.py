import os
from keras.models import load_model
from data_preprocessing import load_and_preprocess_data

def evaluate_model():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    dataset_path = os.path.join(BASE_DIR, 'dataset', 'covid_data.csv')

    model_path = os.path.join(BASE_DIR, 'models', 'ann_covid_model.h5')
    result_path = os.path.join(BASE_DIR, 'results', 'accuracy_report.txt')

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            "Model not found. Please run train_model.py first."
        )

    X_train, X_test, y_train, y_test = load_and_preprocess_data(dataset_path)

    model = load_model(model_path)
    loss, accuracy = model.evaluate(X_test, y_test)

    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    with open(result_path, 'w') as f:
        f.write(f"Test Accuracy: {accuracy * 100:.2f}%\n")

    print(f"Test Accuracy: {accuracy * 100:.2f}%")

if __name__ == "__main__":
    evaluate_model()
