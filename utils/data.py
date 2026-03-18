from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_breast_cancer_data(test_size=0.2, random_state=42):
    data = load_breast_cancer()

    X = data.data.astype("float32")
    y = data.target.astype("float32")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state
    )

    return X_train, X_test, y_train, y_test


def prepare_tabular_data():
    X_train, X_test, y_train, y_test = load_breast_cancer_data()

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, scaler