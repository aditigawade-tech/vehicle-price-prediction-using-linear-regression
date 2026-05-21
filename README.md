# 🚗 Vehicle Price Prediction using Linear Regression

## 📌 Problem Statement
Create a Linear Regression model to predict the price (**dependent variable**) of a vehicle using various **independent parameters**.

---

## 📌 Project Overview
This project demonstrates how to build a **Linear Regression Machine Learning model** to predict vehicle prices using different features such as:

- Brand
- Year
- Fuel Type
- Transmission
- Kilometers Driven
- And other vehicle-related parameters

### 🎯 Main Objectives
- Data Preprocessing
- Feature Selection
- Model Training
- Prediction using Linear Regression
- Model Evaluation

---

## 📌 Technologies Used
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook

---

# 📌 Steps to Create the Linear Regression Model

## 1️⃣ Import Required Libraries

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
```

---

## 2️⃣ Load the Dataset

```python
df = pd.read_csv("car_data.csv")
```

---

## 3️⃣ Explore the Dataset

```python
df.head()
df.info()
df.describe()
```

---

## 4️⃣ Check Missing Values

```python
df.isnull().sum()
```

---

## 5️⃣ Data Preprocessing

Convert categorical columns into numerical values using encoding.

```python
df = pd.get_dummies(df, drop_first=True)
```

---

## 6️⃣ Define Independent and Dependent Variables

### Independent Variables (X)
Features used for prediction.

### Dependent Variable (y)
Vehicle selling price.

```python
X = df.drop("Selling_Price", axis=1)
y = df["Selling_Price"]
```

---

## 7️⃣ Split the Dataset

Divide data into training and testing sets.

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

---

## 8️⃣ Train the Linear Regression Model

```python
model = LinearRegression()

model.fit(X_train, y_train)
```

---

## 9️⃣ Make Predictions

```python
y_pred = model.predict(X_test)
```

---

## 🔟 Evaluate the Model

```python
print("MAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))
```

---

# 🚀 Future Improvements

- Use advanced algorithms like:
  - Random Forest Regression
  - XGBoost
  - Decision Tree Regression
- Hyperparameter Tuning
- Feature Scaling
- Improve Prediction Accuracy

