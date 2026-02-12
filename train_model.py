import pandas as pd
import pickle
from sklearn.ensemble import RandomForestRegressor

# Load dataset
df = pd.read_csv("salary_data.csv")

# Clean columns
df.columns = df.columns.str.strip().str.lower()

# Split features and target
X = df.drop("salary", axis=1)
y = df["salary"]

# Convert categorical columns
X = pd.get_dummies(X)

# Train model
model = RandomForestRegressor()
model.fit(X, y)

# Save model
with open("salary_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model created successfully!")