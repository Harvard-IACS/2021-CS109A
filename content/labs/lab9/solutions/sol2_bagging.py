#1
def get_bagging_df(x_train, y_train, model, bootstrap_range, cv_samples = 5):
    """
    This function will take a model and a range of values and return a dataframe with columns: ["depth", "cross_val_acc"]
    
    Arguments:
        model: the model to be run. Specifically this should be a class instance such as
            DecisionTreeClassifier().
        bootstrap_range: the range of bootstraps ensembles over which BaggingClassifier model's will be fit. 
                        Specifically the BaggingClassifier will fit this number of Decision Trees to get a prediction.
        
    """
    #write an assert statement that enforces that the model entered is a BaggingClassifier
    assert type(model) == BaggingClassifier

    #declare dictionaries
    mean_CV_acc = {}
    all_CV_acc = {}
    
    #find and store cross_validated scores
    for n_estimators in list(bootstrap_range):
        
        model.n_estimators = n_estimators

        score = cross_val_score(estimator=model, X=x_train, y=y_train, n_jobs=-1, cv= cv_samples)
        
        all_CV_acc[n_estimators] = list(score) 
        mean_CV_acc[n_estimators] = score.mean() 
    
    #make a dataframe from the dictionary:
    cv_acc_pd = pd.melt(pd.DataFrame(all_CV_acc))
    cv_acc_pd.columns = ["n_estimators", "cv_acc_score"]
    return cv_acc_pd

#2
bagging_val_acc = get_bagging_df(x_train, 
                        y_train, 
                        BaggingClassifier(base_estimator = DecisionTreeClassifier(max_depth = DEPTH)), 
                        bootstrap_range = range(1,20),
                        cv_samples = 5
                        )
bagging_val_acc.head()

bagging_mean_acc  = bagging_val_acc.groupby("n_estimators").mean()