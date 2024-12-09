#!/usr/bin/env python
# coding: utf-8

# # Model 1 using MLP Neural Network

# In[38]:


# Name: Anthony Tan Beng Wah   NRIC: 095C
# Import the requried packages
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import sqlite3
import requests
import os
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn import linear_model, neighbors, tree, svm, ensemble, neural_network
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, confusion_matrix, classification_report
from sklearn.model_selection import cross_val_score, cross_validate


# In[39]:


# Data Retrieval from Database
# URL of the SQLite database
url = "https://techassessment.blob.core.windows.net/aiap19-assessment-data/agri.db"
file_name = "agri.db"

# Download the file from the URL
if not os.path.exists(file_name):  # Check if the file is already downloaded
    print(f"Downloading {file_name}...")
    response = requests.get(url)
    with open(file_name, 'wb') as file:
        file.write(response.content)
    print("Download complete.")

# Connect to the SQLite database
conn = sqlite3.connect(file_name)

# List tables in the database
query = "SELECT name FROM sqlite_master WHERE type='table';"
tables = pd.read_sql_query(query, conn)
print("Tables in the database:")
print(tables)

# Read data from farm_data table and load into agri dataframe
table_name = "farm_data"  # Replace with the table name you want to query
agri = pd.read_sql_query(f"SELECT * FROM {"farm_data"};", conn)

# Display the data
print(agri.head())


# In[40]:


# display the datatype of agri db attributes
agri.info()


# In[41]:


agri = agri.drop(columns=['System Location Code'])   # This column is not necessary and can be dropped


# In[42]:


# check for columns with null value
agri.isnull().sum()


# In[43]:


# Data Cleansing
# Remove 'ppm' from the nutrient_n, nutrient_p and nutrient_k columns and then convert these 3 columns to float data type
agri['Nutrient N Sensor (ppm)'] = agri['Nutrient N Sensor (ppm)'].str.replace('ppm', '', regex=False).str.strip()
agri['Nutrient N Sensor (ppm)'] = agri['Nutrient N Sensor (ppm)'].astype(float)
agri['Nutrient P Sensor (ppm)'] = agri['Nutrient P Sensor (ppm)'].str.replace('ppm', '', regex=False).str.strip()
agri['Nutrient P Sensor (ppm)'] = agri['Nutrient P Sensor (ppm)'].astype(float)
agri['Nutrient K Sensor (ppm)'] = agri['Nutrient K Sensor (ppm)'].str.replace('ppm', '', regex=False).str.strip()
agri['Nutrient K Sensor (ppm)'] = agri['Nutrient K Sensor (ppm)'].astype(float)


# Replace NaN values with the median value of the column
agri['Nutrient N Sensor (ppm)'] = agri['Nutrient N Sensor (ppm)'].fillna(agri['Nutrient N Sensor (ppm)'].median())
agri['Nutrient P Sensor (ppm)'] = agri['Nutrient P Sensor (ppm)'].fillna(agri['Nutrient P Sensor (ppm)'].median())
agri['Nutrient K Sensor (ppm)'] = agri['Nutrient K Sensor (ppm)'].fillna(agri['Nutrient K Sensor (ppm)'].median())
agri['Temperature Sensor (°C)'] = agri['Temperature Sensor (°C)'].fillna(agri['Temperature Sensor (°C)'].median())
agri['Humidity Sensor (%)'] = agri['Humidity Sensor (%)'].fillna(agri['Humidity Sensor (%)'].median())
agri['Light Intensity Sensor (lux)'] = agri['Light Intensity Sensor (lux)'].fillna(agri['Light Intensity Sensor (lux)'].median())
agri['Water Level Sensor (mm)'] = agri['Water Level Sensor (mm)'].fillna(agri['Water Level Sensor (mm)'].median())

agri.head(20)


# In[44]:


agri.isnull().sum()  # check to confirm no more null value


# In[45]:


# Column values to convert to lowercase for Plant Type and Plant Stage for consistency in data analytics and predictions
agri['Plant Type'] = agri['Plant Type'].str.lower()
agri['Plant Stage'] = agri['Plant Stage'].str.lower()


# In[46]:


# Concatenate Plant Type and Plant Stage and then add the new column into dataframe.  Next, drop the individual Plant Type and Plant Stage columns.
agri['Plant Info'] = agri['Plant Type'] + ' - ' + agri['Plant Stage']
agri = agri.drop(columns=['Plant Type', 'Plant Stage'])
agri.head(20)


# In[47]:


# Set the "temperature" as target/model output and the rest features as model inputs
y = agri['Temperature Sensor (°C)']
X = agri.drop(['Temperature Sensor (°C)'], axis=1)


# In[48]:


# Split the data into training and testing data
from sklearn.model_selection import train_test_split
X_train1, X_test1, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

# Scale down the numerical data (Not Required) and encode the categorical data
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

num_attribs = list(X.select_dtypes(['int64', 'float64']))
cat_attribs = list(X.select_dtypes(['object']))

# Ensure all categorical columns have uniform string types
X_train1[cat_attribs] = X_train1[cat_attribs].astype(str)
X_test1[cat_attribs] = X_test1[cat_attribs].astype(str)

scaler = StandardScaler()
encoder = OrdinalEncoder()

X_train = X_train1.copy()
X_train[num_attribs] = scaler.fit_transform(X_train1[num_attribs])
X_train[cat_attribs] = encoder.fit_transform(X_train1[cat_attribs])

X_test = X_test1.copy()
X_test[num_attribs] = scaler.transform(X_test1[num_attribs])
X_test[cat_attribs] = encoder.transform(X_test1[cat_attribs])


# In[49]:


# Create the mlp (Neural Network) model
mlp_reg = neural_network.MLPRegressor(activation ='relu', hidden_layer_sizes=(10,),
                    max_iter= 2000, solver = 'adam')
# Fit the model to the training set
mlp_reg.fit(X_train,y_train)


# In[50]:


# show the mean absolute error
print('train_mae: ', mean_absolute_error(mlp_reg.predict(X_train), y_train ),
'\n test_mae: ', mean_absolute_error(mlp_reg.predict(X_test), y_test))


# In[51]:


# show the score
print('train_reg:', mlp_reg.score(X_train, y_train),
      '\n test_reg:', mlp_reg.score(X_test, y_test))


# In[52]:


# Find the best # of hidden nodes
hidden_range = range(1,40,3)

train_regression = []
test_regression = []

for h in hidden_range:
    mlp_reg = neural_network.MLPRegressor(activation ='relu', hidden_layer_sizes=(h,),
                    max_iter= 2000, solver = 'adam')
    mlp_reg.fit(X_train, y_train)
    train_regression.append(mlp_reg.score(X_train, y_train))
    test_regression.append(mlp_reg.score(X_test, y_test))

#Plot the train & test accuracy
plt.plot(hidden_range, train_regression, 'bo-', label ='training regression')
plt.plot(hidden_range, test_regression, 'ro-', label = 'testing regression')

plt.xlabel('Hidden Nodes', fontsize='x-large')
plt.ylabel('Accuracy', fontsize='x-large')

plt.legend(loc='best', shadow=True, fontsize='x-large')
plt.show()


# In[53]:


# Final mlp model
mlp_reg = neural_network.MLPRegressor(activation ='relu', hidden_layer_sizes=(31,),
                    max_iter= 2000, solver = 'adam')
mlp_reg.fit(X_train,y_train)


# In[54]:


print('train_mae: ', mean_absolute_error(mlp_reg.predict(X_train), y_train ),
'\n test_mae: ', mean_absolute_error(mlp_reg.predict(X_test), y_test))


# In[55]:


print('train_reg:', mlp_reg.score(X_train, y_train),
      '\n test_reg:', mlp_reg.score(X_test, y_test))


# In[56]:


mlp_reg.score(X_test, y_test)


# In[57]:


test_data = X_test1[0:10]    # Pick 10 rows of test data to show predicted temperatures
test_data


# In[58]:


predicted_temp = y_test[0:10]  # Predicted Temperatures
predicted_temp


# In[59]:


# Build a Machine Learning Pipeline for Model 1 MLP

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

num_attribs = list(X.select_dtypes(['int64','float64']))
cat_attribs = list(X.select_dtypes(['object']))

# Define categorical pipeline
cat_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OrdinalEncoder())
])

# Define numerical pipeline
num_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])


# In[60]:


# Combine categorical and numerical pipelines
preprocessor = ColumnTransformer([
    ('cat', cat_pipe, cat_attribs),
    ('num', num_pipe, num_attribs)
])


# In[61]:


# Fit a pipeline with transformers and an estimator to the training data

pipe = Pipeline([
    ('preprocessor', preprocessor),
    ('model', neural_network.MLPRegressor(activation ='relu', hidden_layer_sizes=(31,),
                    max_iter= 2000, solver = 'adam')),
])


# In[62]:


# Train the pipeline and evaluate the pipeline performance
pipe.fit(X_train, y_train)


# In[63]:


# Save the pipeline model
import joblib
joblib.dump(pipe, "pipe_model1_mlp.pkl")


# In[64]:


# Test the model 1 mlp pipeline with 5 rows of Test Data
some_data = X_test[0:5]
some_data


# In[65]:


my_pipeline_loaded = joblib.load("pipe_model1_mlp.pkl")
my_pipeline_loaded.predict(some_data)


# In[ ]:




