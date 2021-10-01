
# Data and Stats packages
import numpy as np
import pandas as pd

# Visualization packages
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

import warnings

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

def add_higher_order_polynomial_terms(df, N=7):
    df = df.copy()
    cols = df.columns.copy()
    for col in cols:
        for i in range(2, N+1):
            df['{}^{}'.format(col, i)] = df[col]**i
    return df

def make_plots(train_df, val_df, test_df, scaler = None, max_degree = 20):
    """ Make plots of the training and validation dataframes.
    arguments:
        train_df: the training dataframe
        val_df: the validation dataframe
    """
    #preprocessing

    if scaler:
        train_std, scaler = standardized_df(train)
        #val_std = standardized_df(test, scaler) 
        #test_std, scaler = standardized_df(test, scaler) 

    train = train_df
    val = val_df
    test = test_df
    train = train.sort_values(by = ["x1"])
    val = val.sort_values(by = ["x1"])
    test = test.sort_values(by = ["x1"])

    X_train, y__train = train[["x1", "x2", "x3", "x4"]], train["y"]
    X_val, y__val = val[["x1", "x2", "x3", "x4"]], val["y"]
    X_test, y__test = test[["x1", "x2", "x3", "x4"]], test["y"]


    
    
    fig, ax = plt.subplots(1,1, figsize = (16,4))
    ax.set_title("Polynomial regression")
    sns.scatterplot(x = "x1", y = "y", data = train, label = "train")
    sns.scatterplot(x = "x1", y = "y", data = val, label = "validation")
    sns.scatterplot(x = "x1", y = "y", data = test, label = "test")

    for i in range(max_degree):
        #Here I initialize lists inside the for loop in order to emphasize that these lists are used in the context of 
        #this loop.
        if not i:
            train_MSE_lst = []
            val_MSE_lst = []
            test_MSE_lst = []
            train_r2_scores = []
            val_r2_scores = []
            test_r2_scores = []
            lreg_coefs = []

        #poly_df_train = (---) #add_higher_order_polynomial_terms(X_train, N=i)
        #poly_df_test  = (---) #add_higher_order_polynomial_terms(X_test, N=i)

        poly_df_train = add_higher_order_polynomial_terms(X_train, N=i)
        poly_df_val  = add_higher_order_polynomial_terms(X_val, N=i)
        poly_df_test  = add_higher_order_polynomial_terms(X_test, N=i)

        lreg = LinearRegression()
        lreg.fit(poly_df_train, y__train)
        lreg_coefs += list(lreg.coef_)

        #Predict the response variable for the test set
        y_pred_val  = lreg.predict(poly_df_val)
        y_pred_train = lreg.predict(poly_df_train)
        y_pred_test = lreg.predict(poly_df_test)

        #Compute the MSE
        train_MSE = mean_squared_error(y_true = y__train, y_pred = y_pred_train)
        val_MSE = mean_squared_error(y_true = y__val, y_pred = y_pred_val)
        test_MSE = mean_squared_error(y_true = y__test, y_pred = y_pred_test)

        train_r2 = r2_score(y_true = y__train, y_pred = y_pred_train)
        val_r2 = r2_score(y_true = y__val, y_pred = y_pred_val)
        test_r2 = r2_score(y_true = y__test, y_pred = y_pred_test)

        plt.plot( train["x1"], y_pred_train, alpha = 0.05, linewidth = 3, color = "red")

        train_MSE_lst.append(train_MSE)
        val_MSE_lst.append(val_MSE)
        test_MSE_lst.append(test_MSE)

        train_r2_scores.append(train_r2)
        val_r2_scores.append(val_r2)
        test_r2_scores.append(test_r2)

    plt.show()
    data = pd.DataFrame({"train_mse" : train_MSE_lst, 
                         "val_mse" : val_MSE_lst, 
                         "test_mse" : test_MSE_lst,
                         "train_r2_score" : train_r2_scores,
                         "val_r2_score" : val_r2_scores,
                         "test_r2_score" : test_r2_scores,
                         "degree" : list(range(len(train_MSE_lst)))})

    fig, ax = plt.subplots(1,2, figsize = (16,5))

    sns.lineplot(x = "degree", y = "train_mse", data = data, label = "Train", ax = ax[0])
    sns.lineplot(x = "degree", y = "val_mse", data = data, label = "Validation", ax = ax[0])
    sns.lineplot(x = "degree", y = "test_mse", data = data, label = "Test", ax = ax[0])
    ax[0].set_title("Polynomial Linear Regression: degree vs MSE")
    ax[0].set_ylabel("MSE")

    sns.lineplot(x = "degree", y = "train_r2_score", data = data, label = "Train", ax = ax[1])
    sns.lineplot(x = "degree", y = "val_r2_score", data = data, label = "Validation", ax = ax[1])
    sns.lineplot(x = "degree", y = "test_r2_score", data = data, label = "Test", ax = ax[1])
    ax[1].set_title("Polynomial Linear Regression: degree vs r2_score")
    ax[1].set_ylabel("R^2");
    plt.show()

    #sns.displot(lreg_coefs, kde=True)
    plt.hist(lreg_coefs, bins = 40)
    plt.title("displot of beta coefficients")
    best_degree_idx = pd.Series(val_r2_scores).idxmax()
    
    return test_r2_scores[best_degree_idx]

# 1.1 Complete a function that returns a new pandas dataframe
#***Conceptual question: why do we capitalize the X in X_train but not the y in y_train? Briefly discuss this with your classmates***
def standardized_df(df, standard_scaler = None, verbose = True):
    """takes a dataframe as input and returns a standardized dataframe.
    """
    
    #hint: use the fit_transform of the Standard Scaler class from sklearn to normalize the dataset.
    
    #new_array = #TODO
    new_array = standard_scaler.fit_transform(df)
    
    #StandardScaler().fit_transform(df)
    new_df = pd.DataFrame(new_array)
    
    #MAKE THIS INVISIBLE THIS IS THE CHECK
    try:
        #checks if your solution is correct
        #check if the mean is equal to 0
        assert np.isclose(np.mean(new_array[:,0]), 0, atol = 10**-15)
        
        #check if the standard deviation is equal to 1
        assert np.isclose(np.std(new_array[:,0]), 1, atol = 10**-15)
        if verbose:
            print("your solution is correct")
    except:
        if verbose:
            print("your solution is incorrect")
    
    #label the columns
    new_df.columns = ["x", "y"]
    new_df = new_df.sort_values(by = ["x"])
    
    return new_df
