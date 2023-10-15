import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import csv

# Load data from the CSV file
csv_file = 'using.csv'  # Replace with the path to your CSV file
data = pd.read_csv(csv_file)

# Split the data into features (symptoms) and target (disease labels)
X = data.drop(columns="label")

y = data['label']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.04, random_state=42)

# Create a Decision Tree Classifier
clf = DecisionTreeClassifier()

# Train the classifier
clf.fit(X_train, y_train)

# Make predictions on test data
y_pred = clf.predict(X_test)

# Calculate accuracy (for evaluation)
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

import matplotlib.pyplot as plt

# Predict a disease for new symptoms
def prediction():
    new_symptoms = [[0,0,1,0,0,1,0,0,1,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0]]
    predicted_disease = clf.predict(new_symptoms)
    print(predicted_disease[0])
prediction()

