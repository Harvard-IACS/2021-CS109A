def preprocess(df, standardize = False):
    """Splits the data into training and validation sets.
    arguments:
        df: the dataframe of training and test data you want to split.
        standardize: if True returns standardized data.
    """
    #split the data
    train, test = train_test_split(df, train_size=0.8, random_state = 42)
    
    #sort the data
    train = train.sort_values(by = ["x1"])
    test = test.sort_values(by = ["x1"])

    train.describe()

    X_train, y_train = train[["x1"]], train["y"]
    X_test, y_test = test[["x1"]], test["y"]

    X_train_N = add_higher_order_polynomial_terms(X_train, N=15)
    X_test_N = add_higher_order_polynomial_terms(X_test, N=15)
    
    if standardize:
        scaler = StandardScaler().fit(X_train_N)
        X_train_N = scaler.transform(X_train_N)
        X_test_N = scaler.transform(X_test_N)
    
    #"X_val" : X_val_N, "y_val" : y_val,
    datasets = {"X_train": X_train_N, "y_train":  y_train,  "X_test" : X_test_N, "y_test": y_test}
    return(datasets)

def fit_ridge_and_lasso_cv(X_train, y_train,  X_test, y_test,  
                           k = None, alphas = [10**9], best_OLS_r2 = best_least_squares_r2 ): #X_val, y_val,
    """ takes in train and validation test sets and reports the best selected model using ridge and lasso regression.
    Arguments:
        X_train: the train design matrix
        y_train: the reponse variable for the training set
        X_val: the validation design matrix
        y_train: the reponse variable for the validation set
        k: the number of k-fold cross validation sections to be fed to Ridge and Lasso Regularization.
    """

    # Let us do k-fold cross validation 
    fitted_ridge = RidgeCV(alphas=alphas, cv = k).fit(X_train, y_train)
    fitted_lasso = LassoCV(alphas=alphas, cv = k).fit(X_train, y_train)
    
    print('R^2 score for our original OLS model: {}\n'.format(best_OLS_r2))

    ridge_a = fitted_ridge.alpha_
    ridge_score = fitted_ridge.score(X_test, y_test)
    print('Best alpha for ridge: {}'.format(ridge_a))
    print('R^2 score for Ridge with alpha={}: {}\n'.format(ridge_a, ridge_score))

    lasso_a = fitted_lasso.alpha_
    lasso_score = fitted_lasso.score(X_test, y_test)
    print('Best alpha for lasso: {}'.format(lasso_a))
    
    print('R^2 score for Lasso with alpha={}: {}'.format(lasso_a, lasso_score))
    
    r2_df = pd.DataFrame({"OLS": best_OLS_r2, "Lasso" : lasso_score, "Ridge" : ridge_score}, index = [0])
    r2_df = r2_df.melt()
    r2_df.columns = ["model", "r2_Score"]
    plt.title("Validation set")
    sns.barplot(x = "model", y = "r2_Score", data = r2_df)
    
    plt.show()