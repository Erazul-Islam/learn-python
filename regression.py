import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Example data
data = {
    "Advertising": [1, 2, 3, 4, 5],
    "Sales": [2, 4, 5, 4, 6]
}
df = pd.DataFrame(data)

# Define X and y
X = df[["Advertising"]]   # independent variable
y = df["Sales"]           # dependent variable

# Create and fit model
model = LinearRegression()
model.fit(X, y)

# Predict sales
df["Predicted_Sales"] = model.predict(X)

# Print slope and intercept
print("Intercept:", model.intercept_)
print("Slope:", model.coef_[0])

# Visualization
plt.scatter(df["Advertising"], df["Sales"], color="blue", label="Actual Data")
plt.plot(df["Advertising"], df["Predicted_Sales"], color="red", label="Regression Line")
plt.xlabel("Advertising Budget ($000)")
plt.ylabel("Sales ($000)")
plt.legend()
plt.show()
