import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_preprocess_data(csv_path):
    df = pd.read_csv(csv_path)

    # Normalize column names
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace('-', '_')
    )

    # Create COVID positive target
    df['covid_positive'] = (
        (df['severity_severe'] == 1) |
        (df['severity_moderate'] == 1)
    ).astype(int)

    # Select input features
    df = df[
        [
            'fever',
            'dry_cough',
            'difficulty_in_breathing',
            'sore_throat',
            'tiredness',
            'pains',
            'contact_yes',
            'gender_male',
            'covid_positive'
        ]
    ]

    # Rename for clarity
    df.rename(columns={
        'dry_cough': 'cough',
        'difficulty_in_breathing': 'shortness_of_breath'
    }, inplace=True)

    # Split features and target
    X = df.drop('covid_positive', axis=1)
    y = df['covid_positive']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Feature scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, scaler
