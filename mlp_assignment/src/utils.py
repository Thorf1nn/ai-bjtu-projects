import numpy as np
from sklearn.datasets import fetch_openml, fetch_california_housing

def to_categorical(y, num_classes=None):
    if not num_classes:
        num_classes = np.max(y) + 1
    y = np.array(y, dtype='int').ravel()
    n = y.shape[0]
    categorical = np.zeros((n, num_classes))
    categorical[np.arange(n), y] = 1
    return categorical

def train_test_split(X, y, test_size=0.2, random_state=None):
    if random_state:
        np.random.seed(random_state)
    p = np.random.permutation(X.shape[0])
    X, y = X[p], y[p]
    split_idx = int(X.shape[0] * (1 - test_size))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    return X_train, X_test, y_train, y_test

class StandardScaler:
    def __init__(self):
        self.mean_ = None
        self.scale_ = None

    def fit(self, X):
        self.mean_ = np.mean(X, axis=0)
        self.scale_ = np.std(X, axis=0)
        return self

    def transform(self, X):
        return (X - self.mean_) / self.scale_

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

def load_mnist_data():
    """
    Loads the MNIST dataset.
    """
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
    X = mnist.data.astype('float32')
    y = mnist.target.astype('int')
    
    # Normalize pixel values to be between 0 and 1
    X /= 255.0
    
    # One-hot encode labels
    y = to_categorical(y, 10)
    
    return X, y

def load_california_housing_data():
    """
    Loads the California Housing dataset for regression.
    """
    housing = fetch_california_housing(as_frame=False)
    X = housing.data
    y = housing.target.reshape(-1, 1) # Reshape for consistency
    return X, y

def load_mnist_from_csv(data_path='../data'):
    """
    Loads the MNIST dataset from local CSV files.
    """
    import pandas as pd
    
    # Load training data
    train_df = pd.read_csv(f"{data_path}/mnist_train.csv")
    y_train = train_df['label'].values
    X_train = train_df.drop('label', axis=1).values.astype('float32')

    # Load test data
    test_df = pd.read_csv(f"{data_path}/mnist_test.csv")
    y_test = test_df['label'].values
    X_test = test_df.drop('label', axis=1).values.astype('float32')
    
    # Normalize pixel values
    X_train /= 255.0
    X_test /= 255.0
    
    # One-hot encode labels
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    
    return (X_train, y_train), (X_test, y_test)
