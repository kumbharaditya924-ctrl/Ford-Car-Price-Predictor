import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# Dummy data for training
data = {
    'model': ['Fiesta', 'Focus', 'Fiesta', 'Kuga', 'EcoSport'],
    'year': [2018, 2019, 2017, 2020, 2018],
    'mileage': [30000, 20000, 45000, 15000, 25000],
    'tax': [145, 145, 150, 145, 145],
    'mpg': [55.4, 60.2, 50.1, 47.9, 52.3],
    'engineSize': [1.0, 1.5, 1.2, 2.0, 1.0],
    'transmission': ['Manual', 'Automatic', 'Manual', 'Semi-Auto', 'Manual'],
    'fuelType': ['Petrol', 'Diesel', 'Petrol', 'Diesel', 'Petrol'],
    'price': [12000, 15000, 9500, 22000, 11000]
}
df = pd.DataFrame(data)

X = df.drop(columns=['price'])
y = df['price']

num_cols = ['year', 'mileage', 'tax', 'mpg', 'engineSize']
scaler = StandardScaler()
X[num_cols] = scaler.fit_transform(X[num_cols])
X_encoded = pd.get_dummies(X)

model = LinearRegression()
model.fit(X_encoded, y)

# Creating the 3 pkl files
joblib.dump(model, 'LR_model.pkl')          
joblib.dump(scaler, 'scaler.pkl')          
joblib.dump(X_encoded.columns.tolist(), 'columns.pkl') 

print("🔥 Done! All 3 files created successfully!")