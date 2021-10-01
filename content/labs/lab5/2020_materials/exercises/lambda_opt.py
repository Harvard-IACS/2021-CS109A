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
    new_df.columns = df.columns
    new_df = new_df.sort_values(by = ["x1"])
    
    return new_df

train_, test = train_test_split(df1, train_size=0.8, random_state = 42)
train, val = train_test_split(train_, train_size=0.8, random_state = 42)
scaler = StandardScaler().fit(train)
train_std = standardized_df(train, scaler)

val_std = standardized_df(val, scaler) 
test_std = standardized_df(test, scaler) 
X_train, y_train = train_std[["x1","x2","x3","x4"]], train_std["y"]
X_val, y_val     = val_std[["x1","x2","x3","x4"]], val_std["y"]
X_test, y_test   = test_std[["x1","x2","x3","x4"]], test_std["y"]

#Assume that N =12. Optimize for lambda, the ridge regularization hyper parameter:
poly_df_train = add_higher_order_polynomial_terms(X_train, N=5)
poly_df_val  = add_higher_order_polynomial_terms(X_val, N=5)
poly_df_test  = add_higher_order_polynomial_terms(X_test, N=5)


fig, ax = plt.subplots(1,1, figsize = (16,4))
sns.scatterplot(x = "x1", y = "y", data = train_std, label = "train")
sns.scatterplot(x = "x1", y = "y", data = val_std, label = "test")




for i, lambda_exponent in enumerate(range(12, 20)):
    if not i:
        train_MSE_lst = []
        test_MSE_lst = []
        train_r2_scores = []
        test_r2_scores = []
        ridge_coefs = []
        log_lambda_list = []
    lambda_ = 10**(lambda_exponent)
    log_lambda_list.append(lambda_exponent)
    
    ridge = Ridge(alpha=lambda_) ##
    ridge.fit(poly_df_train, y_train)
    ridge_coefs += list(ridge.coef_.flatten())

    #Predict the response variable for the test set
    y_pred_val  = ridge.predict(poly_df_val)
    y_pred_train = ridge.predict(poly_df_train)
    y_pred_test = ridge.predict(poly_df_test)

    #Compute the MSE
    train_MSE = mean_squared_error(y_true = y_train, y_pred = y_pred_train)
    test_MSE = mean_squared_error(y_true = y_val, y_pred = y_pred_val)
    
    train_r2 = r2_score(y_true = y_train, y_pred = y_pred_train)
    test_r2 = r2_score(y_true = y_val, y_pred = y_pred_val)
    
    plt.plot( train_std["x1"], y_pred_train, alpha = lambda_, linewidth = 3, color = "red")
    
    train_MSE_lst.append(train_MSE)
    test_MSE_lst.append(test_MSE)
    
    train_r2_scores.append(train_r2)
    test_r2_scores.append(test_r2)
plt.show()
data = pd.DataFrame({"train_mse" : train_MSE_lst, 
                     "test_mse" : test_MSE_lst, 
                     "train_r2_score" : train_r2_scores,
                     "test_r2_score" : test_r2_scores,
                     "log10_lambda" : log_lambda_list,
                     "degree" : list(range(len(train_MSE_lst)))})

fig, ax = plt.subplots(1,2, figsize = (16,5))

#sns.lineplot(x = "log10_lambda", y = np.log(data["train_mse"]), data = data, label = "Train", ax = ax[0])
#sns.lineplot(x = "log10_lambda", y = np.log(data["test_mse"]), data = data, label = "Test", ax = ax[0])
sns.lineplot(x = "log10_lambda", y = "train_mse", data = data, label = "Train", ax = ax[0])
sns.lineplot(x = "log10_lambda", y = "test_mse", data = data, label = "Test", ax = ax[0])
ax[0].set_title("Ridge Regression: lambda vs MSE")
ax[0].set_ylabel("MSE")

sns.lineplot(x = "log10_lambda", y = "train_r2_score", data = data, label = "Train", ax = ax[1])
sns.lineplot(x = "log10_lambda", y = "test_r2_score", data = data, label = "Test", ax = ax[1])
ax[1].set_title("Ridge Regression: lambda vs r2_score")
ax[1].set_ylabel("R^2");
plt.show()

#sns.displot(lreg_coefs, kde=True)
plt.hist(ridge_coefs, bins = 40)
plt.title("displot of beta coefficients")