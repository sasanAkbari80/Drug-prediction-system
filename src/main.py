import pandas as pd
from sklearn.model_selection import train_test_split

from preprocess import encode_features
from train import train_model
from evaluate import evaluate_model


# Load data
df = pd.read_csv('../data/drug200.csv')

# Features & Target
X = df[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']].values
y = df['Drug'].values

# Preprocess
X = encode_features(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=3
)

# Train
model = train_model(X_train, y_train)

# Evaluate
acc, preds = evaluate_model(model, X_test, y_test)

print("Accuracy:", acc)
print("Sample predictions:", preds[:5])
