import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

# Load dataset
df = pd.read_csv(r"D:\Health insurance prediction\cleaned_insurance.csv")

# Define features and target
X = df.drop(columns=["charges"])
y = (df["charges"] > df["charges"].median()).astype(int)  # Binary classification

# Standardize Features Before PCA
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Dimensionality Reduction with PCA (Ensure it matches 5 features)
pca = PCA(n_components=5)
X_pca = pca.fit_transform(X_scaled)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

# Define models
models = {
    "RandomForest": RandomForestClassifier(),
    "SVM": SVC(),
    "LogisticRegression": LogisticRegression(),
    "DecisionTree": DecisionTreeClassifier(),
    "KNN": KNeighborsClassifier()
}

# Train and evaluate models
best_model = None
best_score = 0
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"{name} - Accuracy: {acc:.2f}, Precision: {prec:.2f}, Recall: {recall:.2f}, F1 Score: {f1:.2f}")
    
    if f1 > best_score:
        best_score = f1
        best_model = model

# Hyperparameter Tuning (Randomized Search)
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 20, None],
    'min_samples_split': [2, 5, 10]
}

rf_random = RandomizedSearchCV(RandomForestClassifier(), param_grid, n_iter=20, cv=5, random_state=42, n_jobs=-1)
rf_random.fit(X_train, y_train)
best_model = rf_random.best_estimator_

# Save best model, scaler, and PCA
joblib.dump(best_model, "best_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(pca, "pca.pkl")

print("Model, scaler, and PCA saved successfully!")
