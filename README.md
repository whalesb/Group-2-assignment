# 📊 Machine Learning Analysis with Python

## 📌 Introduction
This Jupyter Notebook explores how long it takes for a person to become a professional after starting to code. The dataset is analyzed using **Pandas**, **Matplotlib**, **Seaborn**, and **Scikit-learn** for data visualization and machine learning.

---

## 📂 Required Libraries
```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
```

---

## 📥 Loading and Exploring the Dataset
```python
df = pd.read_csv("C:/Users/Wale/Documents/class machine learning/Group_2_Data_set.csv")
df.head()
```
- Loads the dataset into a Pandas DataFrame.
- Displays the first five rows.

```python
df.info()
```
- Displays dataset metadata (column names, data types, and missing values).

```python
len(df)
```
- Returns the total number of rows in the dataset.

---

## 🔍 Data Cleaning and Preprocessing
### **1️⃣ Selecting the First 50 Rows**
```python
df = df.iloc[:50, :]
len(df)
```

### **2️⃣ Handling Missing Data**
```python
Data = df.dropna()
len(Data)
```
- Removes rows containing missing values.

### **3️⃣ Converting Data Types**
```python
DataFrame = Data.astype("int64")
```
- Converts all columns from `object` to `int64`.

### **4️⃣ Adding a New Feature**
```python
DataFrame["How_long_it_took_to_become_a_pro"] = DataFrame['YearsCode'] - DataFrame['YearsCodePro']
```
- Calculates how long it took for each person to become a professional.

---

## 📊 Data Visualization
```python
plt.figure()
sns.regplot(x=DataFrame['Age1stCode'], y=DataFrame['How_long_it_took_to_become_a_pro'],
            scatter_kws={"color": "blue", "alpha": 0.7}, line_kws={"color": "red"})
plt.xlabel("How long it took to become a pro", fontweight="bold")
plt.ylabel("Age I started coding", fontweight="bold")
plt.title("From Beginner to Pro: How Long Does It Take", fontweight="bold")
plt.show()
```
- Creates a regression plot to visualize the relationship between `Age1stCode` and `How_long_it_took_to_become_a_pro`.

---

## 🤖 Machine Learning Model
### **1️⃣ Splitting Data**
```python
X = DataFrame[['Age1stCode']]
y = DataFrame[['How_long_it_took_to_become_a_pro']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```
- Splits data into **training (80%)** and **testing (20%)** sets.

### **2️⃣ Training the Model**
```python
model = LinearRegression()
model.fit(X_train, y_train)
```
- Fits the **Linear Regression Model** to the training data.

### **3️⃣ Making Predictions**
```python
Y_pred = model.predict(X_test)
```
- Predicts `How_long_it_took_to_become_a_pro` for the test set.

---

## 📏 Model Evaluation
```python
MSE = mean_squared_error(y_test, Y_pred)
MAE = mean_absolute_error(y_test, Y_pred)
R2_Score = r2_score(y_test, Y_pred)

print('Mean Squared Error:', MSE)
print('Mean Absolute Error:', MAE)
print('R2 Score:', R2_Score)
```
### **📊 Output:**
```
Mean Squared Error: 4.029
Mean Absolute Error: 1.888
R2 Score: 0.38
```
- The model explains **38%** of the variance in `How_long_it_took_to_become_a_pro`.

---

## 🧠 Predicting for a New Age Input
```python
age = 15
How_long_will_it_take_me_to_become_a_pro = model.predict([[age]])
print("If I start coding at age", age, "I will become a pro in", How_long_will_it_take_me_to_become_a_pro, "years")
```
### **📊 Output:**
```
If I start coding at age 15, I will become a pro in [[5.71]] years
```
- If someone starts coding at **15**, they will become a pro in **~6 years**.

---

## 🚀 Conclusion
This project analyzes how long it takes to become a professional in coding using Python and **Linear Regression**. The model, though not perfect, provides insights into learning duration trends.

---

## 🏗 Future Improvements
- Collect a **larger dataset** for better accuracy.
- Consider **multiple regression** using more variables.
- Experiment with **advanced ML models**.

📢 **Feel free to contribute, fork, or improve this project!** 🚀

