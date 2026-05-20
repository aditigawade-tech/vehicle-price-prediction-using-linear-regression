##Title -Vehicle Price Prediction using Linear Regression
•	##Problem statement -crete a linear regression model To predict price(dependent variable) of vehicle by using various independent parameters

📌## Project Overview
This project demonstrates how to build a Linear Regression Machine Learning model to predict the price of a vehicle (dependent variable) using different independent parameters such as brand, year, fuel type, transmission, kilometers driven, etc.
The main goal of this project is to understand:
•	Data preprocessing
•	Feature selection
•	Model training
•	Prediction using Linear Regression
•	Model evaluation

📌 ##Technologies Used
•	Python
•	Pandas
•	NumPy
•	Matplotlib
•	Seaborn
•	Scikit-learn
•	Jupyter Notebook

📌## Steps to Create the Linear Regression Model
1️⃣ ##Import Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

2️⃣ ##Load the Dataset
df = pd.read_csv("car_data.csv")

3️⃣ ##Explore the Dataset
df.head()
df.info()
df.describe
4️⃣ ##Check Missing Values
df.isnull().sum()

5️⃣ ##Data Preprocessing
Convert categorical columns into numerical values using encoding.
df = pd.get_dummies(df, drop_first=True)

6️⃣ ##Define Independent and Dependent Variables
Independent Variables (X)
Features used for prediction.
Dependent Variable (y)
Vehicle selling price.
X = df.drop("Selling_Price", axis=1)
y = df["Selling_Price"]

7️⃣ ##Split the Dataset
Divide data into training and testing sets.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

8️⃣ ##Train the Linear Regression Model
model = LinearRegression()

model.fit(X_train, y_train)

9️⃣ ##Make Predictions
y_pred = model.predict(X_test)

🔟 ##Evaluate the Model
print("MAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))



🚀 ##Future Improvements
•	Use advanced algorithms like: 
o	Random Forest Regression 
o	XGBoost 
o	Decision Tree Regression 
•	Hyperparameter tuning 
•	Feature scaling 
•	Improve prediction accuracy 
