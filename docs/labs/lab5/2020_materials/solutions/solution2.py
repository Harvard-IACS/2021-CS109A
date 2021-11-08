def fit_ridge_and_lasso_cv(X_train, y_train,  X_test, y_test,  
                           k = None, alphas = [10**9], best_OLS_r2 = best_OLS_r2): #X_val, y_val,
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