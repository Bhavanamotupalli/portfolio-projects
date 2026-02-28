print("Script is running...")

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Dummy dataset (temporary)
data = pd.DataFrame({
    "tenure": [1, 5, 10, 3, 8, 2, 12, 6],
    "monthly_charges": [50, 70, 80, 40, 90, 60, 100, 75],
    "churn": [1, 0, 0, 1, 0, 1, 0, 0]
})

X = data[["tenure", "monthly_charges"]]
y = data["churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, predictions))
