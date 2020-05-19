# -*- coding: utf-8 -*-
"""

@author: Emmanuel
"""
# pandas, numpy ans seaborne
import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd 
import seaborn as sns

# Sk learn libs used
from sklearn.preprocessing import LabelEncoder 
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score 
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import OrthogonalMatchingPursuitCV
from sklearn.metrics import  mean_squared_error

# for saving models
import joblib
# miscellaneous
import warnings
import time

# function to run our regression models 
def run_reg_models(classifer_names, classifiers, X_train, X_test, y_train, y_test, save = 0, save_index = 0): 
    counter = 0
    for name, clf in zip(classifer_names, classifiers): 
        result = clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        model_performance = pd.DataFrame(data = [r2_score(y_test, y_pred), np.sqrt(mean_squared_error(y_test, y_pred))],
                                 index = ["R2","RMSE"])
        print(name + ' performance: ')
        print(model_performance)
        
        #then visualize the result 
        visualize_results(y_pred, y_test)
        # save one classifier
        if(save == 1): 
            if(counter != save_index): 
                counter += 1
            else:  
                print("i saved: "+ name)              
                joblib.dump(result, name)
                # it's now saved
                save = 0

def handle_cat_data(cat_feats, data): 
    for f in cat_feats: 
        to_add = pd.get_dummies(data[f], prefix= f, drop_first = True)
        merged_list = data.join(to_add, how='left', lsuffix='_left', rsuffix='_right')
        data = merged_list
        
    # then drop the categorical features
    data.drop(cat_feats, axis=1, inplace=True)
    
    return data

def visualize_results(y_pred,y_test): 
    # # so now we drawing a side by side bar chart to compare predicted and actual values 
    df = pd.DataFrame({'Actual':y_test.values.flatten(), 'Predicted':y_pred.flatten()})
    
    # # Plot Bar chart to show the variation between the actual values and the predicted values
    df1 = df.head(25)
    df1.plot(kind='bar',figsize=(16,10))
    plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
    plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
    plt.show()
      
# function to load and run saved models
# To easily run saved models 
def load_and_run_models(models,model_names, X_train, y_train, X_test, y_test): 
    for model,name in zip(models, model_names): 
        start_time = time.time()
        print("Currently on: " + name +'...')
        # load and fit model
        loaded_model = joblib.load(model + '.pkl')
        result = loaded_model.fit(X_train, y_train)
        
        y_pred = result.predict(X_test)
        model_performance = pd.DataFrame(data = [r2_score(y_test, y_pred), np.sqrt(mean_squared_error(y_test, y_pred))],
                                 index = ["R2","RMSE"])
        print(name + ' performance: ')
        print(model_performance)
        # get the time it took to run 
        end_time = time.time()
        print(name +" took: " + str(end_time - start_time)+ " seconds")

  
# clear warnings to make output clearer and easier to read
warnings.filterwarnings('ignore')

df_insurance = pd.read_csv('insurance.csv')

# check if we have any empty values 
# print(df_insurance.isnull().sum())

# categorical features we need to encode with label encoder
cat_features = ['sex', 'smoker', 'region']

# use the handle categorical data function 
# convert categorical data to binary
final_insurance_df = handle_cat_data(cat_features, df_insurance)

final_insurance_vars = final_insurance_df.columns.values

# # # get X features, excluding charges column
X_features = final_insurance_vars[final_insurance_vars != 'charges']

# # # get Y features, charges alone
Y_features = ['charges']
# # # handle class labels
gle = LabelEncoder()

labels = gle.fit_transform(final_insurance_df[Y_features])
final_insurance_df[Y_features] = labels

# divide dataset into training and testing dataset
X_train, X_test, y_train, y_test = train_test_split(final_insurance_df[X_features],final_insurance_df[Y_features], test_size = 0.2, random_state = 0 )

# please note the parameter, normalize to handle categorical data
estimator_names = ['Linear Regression', 'Ridge Regression', 'Lasso Regression', 'Elastic Net Regression', 'Orthongonal Matching Pursuit CV']
estimators = [
                LinearRegression(normalize=True), 
                Ridge(alpha=0, normalize=True), 
                Lasso(alpha=0.01, normalize=True), 
                ElasticNet(random_state=0, normalize=True), 
                OrthogonalMatchingPursuitCV(cv=8, normalize=True)
            ]
# run our estimator models 
# run_reg_models(estimator_names, estimators, X_train, X_test, y_train, y_test, save = 1, save_index = 0)


# # correlation 
# sns.heatmap(final_insurance_df.corr(), annot = True)


# -------------- LOAD ALL SAVED REGRESSION MODELS ------------------------
models_to_load = [
        'Linear Regression', 
    ]

load_and_run_models(models_to_load,models_to_load, X_train, y_train, X_test, y_test )