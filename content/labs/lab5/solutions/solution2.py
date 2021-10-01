datasets = preprocess(df1, standardize = True)
#Here we input arguments to a function from a dictionary using the ** syntax option.
# X_train, y_train, X_val, y_val,

fit_ridge_and_lasso_cv(**datasets, k = 4, alphas = [10**i for i in range(-10, 5)])