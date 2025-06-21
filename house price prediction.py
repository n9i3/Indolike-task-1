#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# In[2]:


column_names = [
    'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX',
    'RM', 'AGE', 'DIS', 'RAD', 'TAX',
    'PTRATIO', 'B', 'LSTAT', 'MEDV'
]


# In[3]:


df = pd.read_csv("housing.csv", delim_whitespace=True, header=None, names=column_names)


# In[4]:


df.head(5)


# In[5]:


df.drop_duplicates(inplace=True)
df.dropna(inplace=True)
df = df[df['CRIM'] < df['CRIM'].quantile(0.99)]


# In[6]:


features = ['RM', 'AGE', 'TAX', 'RAD', 'DIS', 'CRIM']
X = df[features]
y = df['MEDV']



# In[7]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Standardization
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 6. Model training
model = LinearRegression()
model.fit(X_train_scaled, y_train)


# In[8]:


# 7. Predictions and evaluation
y_pred = model.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n📊 Model Evaluation:")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R² Score: {r2:.2f}")



# In[9]:


plt.figure(figsize=(8, 5))
sns.scatterplot(x=y_test, y=y_pred, color='blue', s=60, edgecolor='k')
min_val = min(min(y_test), min(y_pred))
max_val = max(max(y_test), max(y_pred))
plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', label='Perfect Prediction')

plt.title("📊 Actual vs Predicted Housing Prices")
plt.xlabel("Actual MEDV ($1000s)")
plt.ylabel("Predicted MEDV ($1000s)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()  


# In[13]:


import re

print("\n Predict Housing Price Based on Inputs")
try:
    rm = int(input("Enter number of rooms (e.g.3,6,2): "))
    age = float(input("Enter age of the house : "))
    tax = float(input("Enter property tax rate in % : "))
    rad = int(input("Enter RAD (access to highways) [1–24]: "))

    
    dis_input = input("Enter distance to employment centers (e.g., 5 km or 5km): ")
    dis_cleaned = re.sub(r"[^\d.]", "", dis_input)  
    dis = float(dis_cleaned)

    crim = float(input("Enter crime rate per capita (e.g., 0.03): "))

    user_input = pd.DataFrame({
        'RM': [rm],
        'AGE': [age],
        'TAX': [tax],
        'RAD': [rad],
        'DIS': [dis],
        'CRIM': [crim]
    })

    user_scaled = scaler.transform(user_input)
    pred_price = model.predict(user_scaled)[0]

    print(f"\n💰 Estimated House Price: ₹{pred_price * 100000:.2f}")

except Exception as e:
    print(f"⚠️ Error: {e}")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




