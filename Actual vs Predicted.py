import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt


data = pd.DataFrame({
    'Deliveries': [35, 38, 42, 40, 37, 45, 50, 48, 32, 30, 28, 33, 36, 44],
    'Day': ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
    'PublicHoliday': [0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0],
    'Region': ['Gauteng', 'WC', 'WC', 'Gauteng', 'WC', 'Gauteng', 'WC',
               'Gauteng', 'WC', 'Gauteng', 'Gauteng', 'WC', 'WC', 'Gauteng']
})


data_encoded = pd.get_dummies(data, columns=['Day', 'Region'], drop_first=True)


X = data_encoded.drop(columns='Deliveries')  
y = data_encoded['Deliveries']              


model = LinearRegression()
model.fit(X, y)        
predictions = model.predict(X)  


mean_y = np.mean(y)
ss_total = np.sum((y - mean_y) ** 2)         # Total variation in data
ss_residual = np.sum((y - predictions) ** 2) # Unexplained variation
r_squared = 1 - (ss_residual / ss_total)
mse = np.mean((y - predictions) ** 2)


print(" Intercept (baseline prediction):", round(model.intercept_, 2))
print(" Coefficients for each variable:")
for name, coef in zip(X.columns, model.coef_):
    print(f" {name}: {round(coef, 2)}")

print(f"R-squared: {round(r_squared, 3)} (explains {round(r_squared*100)}% of the variation)")
print(f" Mean Squared Error: {round(mse, 2)}")


plt.figure(figsize=(8, 5))
plt.plot(y.values, label='Actual Deliveries', marker='o')
plt.plot(predictions, label='Predicted Deliveries', marker='x')
plt.title('Actual vs Predicted Daily Deliveries')
plt.xlabel('Day Index')
plt.ylabel('Number of Deliveries')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
