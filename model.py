# Import dependencies
import pandas as pd
import numpy as np

from sklearn.externals import joblib as jl


pd.options.mode.chained_assignment = None 

# Load the dataset in a dataframe object 
url = "http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/train.csv"
df = pd.read_csv(url)

# Include only four features
include = ['Age', 'Sex', 'Embarked', 'Survived'] 
df_ = df[include]
print(df)

# Handle missing values and separate/convert non-numeric columns to numeric
categoricals = []
for col, col_type in df_.dtypes.iteritems():
     if col_type == 'O':
          categoricals.append(col)
     else:
          df_[col].fillna(0, inplace=True)
df_ohe = pd.get_dummies(df_, columns=categoricals, dummy_na=True)
print(df_ohe)

# Train the machine learning model with Logistic Regression classifier
from sklearn.linear_model import LogisticRegression
dependent_variable = 'Survived'
x = df_ohe[df_ohe.columns.difference([dependent_variable])]
y = df_ohe[dependent_variable]
lr = LogisticRegression()
lr.fit(x, y)
print(lr)

#Pickle Model - persist the Logistic Regression - Serialize
jl.dump(lr, 'titanic_model.pkl')

#Deserialize the model
lr = jl.load('titanic_model.pkl')
print(lr)

#Export the model columns
model_columns = list(x.columns)
jl.dump(model_columns, 'titanic_model_columns.pkl')
print(model_columns)

