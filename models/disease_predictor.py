import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle

# Load dataset
df = pd.read_csv('data/Disease_symptom_and_patient_profile_dataset.csv')

# Prepare data
le = LabelEncoder()
df['Fever'] = le.fit_transform(df['Fever'])
df['Cough'] = le.fit_transform(df['Cough'])
df['Fatigue'] = le.fit_transform(df['Fatigue'])
df['Difficulty Breathing'] = le.fit_transform(df['Difficulty Breathing'])
df['Gender'] = le.fit_transform(df['Gender'])
df['Blood Pressure'] = le.fit_transform(df['Blood Pressure'])
df['Cholesterol Level'] = le.fit_transform(df['Cholesterol Level'])

# Features and target
X = df[['Fever', 'Cough', 'Fatigue', 'Difficulty Breathing', 'Age', 'Gender', 'Blood Pressure', 'Cholesterol Level']]
y = df['Disease']

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model
with open('models/model.pkl', 'wb') as f:
    pickle.dump(model, f)

print(f"Model trained successfully!")
print(f"Accuracy: {model.score(X_test, y_test) * 100:.2f}%")