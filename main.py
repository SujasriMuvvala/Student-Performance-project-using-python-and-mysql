import mysql.connector
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Connect to MySQL
conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="Muvvala@2007",
    database="student_performance"
)

# Read data from database
query = "SELECT * FROM students"
df = pd.read_sql(query, conn)

# Display student data
print("\nSTUDENT DATA")
print(df)

# -------------------------------
# DATA ANALYSIS
# -------------------------------

average_marks = df['internal_marks'].mean()

print("\nAverage Internal Marks:")
print(average_marks)

top_student = df.loc[df['internal_marks'].idxmax()]

print("\nTop Performing Student:")
print(top_student)

low_attendance = df[df['attendance'] < 75]

print("\nStudents with Low Attendance:")
print(low_attendance)

# -------------------------------
# VISUALIZATION
# -------------------------------

# Bar Chart
plt.figure(figsize=(8,5))

plt.bar(df['name'], df['internal_marks'])

plt.xlabel("Student Names")
plt.ylabel("Internal Marks")
plt.title("Student Marks Analysis")

plt.show()

# Line Chart
plt.figure(figsize=(8,5))

plt.plot(df['name'], df['attendance'], marker='o')

plt.xlabel("Student Names")
plt.ylabel("Attendance Percentage")
plt.title("Attendance Analysis")

plt.show()

# -------------------------------
# MACHINE LEARNING MODEL
# -------------------------------

# Features
X = df[['attendance', 'study_hours',
        'assignment_score', 'internal_marks']]

# Target
y = df['final_result']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create model
model = LogisticRegression()

# Train model
model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, predictions)

print("\nModel Accuracy:")
print(accuracy)

# Example Prediction
sample_student = [[80, 5, 75, 78]]

result = model.predict(sample_student)

print("\nPrediction for New Student:")

if result[0] == 1:
    print("PASS")
else:
    print("FAIL")

# Close connection
conn.close()