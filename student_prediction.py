import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv("student_data.csv")

# Split data
X = data[['Hours_Studied']]
y = data['Exam_Score']

# Train model
model = LinearRegression()
model.fit(X, y)

# Prediction
hours = [[7]]
predicted_score = model.predict(hours)

print("Predicted Exam Score for 7 hours study:", predicted_score[0])

# Plot graph
plt.scatter(X, y)
plt.plot(X, model.predict(X))
plt.xlabel("Hours Studied")
plt.ylabel("Exam Score")
plt.title("Student Performance Prediction")
plt.show()