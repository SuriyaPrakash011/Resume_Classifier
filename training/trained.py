import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pickle
import os

# Step 1: Load cleaned resumes
df = pd.read_csv("data/cleaned_resumes.csv")
print("Sample Data:\n", df.head())

# Step 2: Convert text to TF-IDF features
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['Cleaned_Resume'])
y = df['Category']

# Step 3: Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 4: Train Logistic Regression model
model = LogisticRegression(max_iter=300)
model.fit(X_train, y_train)

# Step 5: Evaluate
y_pred = model.predict(X_test)
print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Step 6: Save model + vectorizer
os.makedirs("models", exist_ok=True)
pickle.dump(model, open("models/resume_model.pkl", "wb"))
pickle.dump(vectorizer, open("models/vectorizer.pkl", "wb"))

print("\n✅ Model and Vectorizer saved in 'models/' folder")
