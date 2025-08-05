from tabpfn_client import TabPFNClassifier, init
import tabpfn_client
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
init()
import tabpfn_client
# Get your token
token = tabpfn_client.get_access_token()

tabpfn_client.set_access_token(token)
# Load the Iris dataset
X, y = load_iris(return_X_y=True)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train classifier
classifier = TabPFNClassifier(device='cuda', N_ensemble_configurations=10)
classifier.fit(X_train, y_train)

# Evaluate
y_pred = classifier.predict(X_test)
print('Test Accuracy:', accuracy_score(y_test, y_pred))