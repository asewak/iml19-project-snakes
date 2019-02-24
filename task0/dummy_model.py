
import sys
sys.path.append("/Users/ainesh/Documents/ETH Sem 1/Data Mining 1/Assignment 4/homework4/")

import numpy as np 
import pandas as pd
import math
from sklearn.metrics import mean_squared_error



input_dir = '/Users/ainesh/Documents/ETH Sem 2/Introduction to Machine Learning/Projects/task0_sl19d1/'
output_dir = '/Users/ainesh/Documents/ETH Sem 2/Introduction to Machine Learning/Projects/task0_sl19d1/'
train_file = input_dir + 'train.csv'
test_file = input_dir + 'test.csv'
out_file = input_dir + 'preds.csv'

df_train = pd.read_csv(train_file)
df_test = pd.read_csv(test_file)

train_cols = list(df_train.columns.values)
x_names = []

# remove Id or y variable
for e in train_cols:
    if e not in ['Id','y']:
        x_names.append(e)


# dummy model  -  mean of x variables
y_pred = df_train.loc[:,x_names].mean(axis=1)

# actuals
y = df_train['y']

# train error
RMSE = mean_squared_error(y, y_pred)**0.5

# test predictions
y_test = df_test.loc[:,x_names].mean(axis=1)
preds = pd.concat([df_test['Id'], y_test], ignore_index=True, axis=1)
preds.columns = ['Id', 'y']

preds.to_csv(out_file, index=False)