def get_impurity_pd(model, n=0):
    """
    This function returns a pandas dataframe with all of the nth nodes feature impurities.

    Arguments:
        model: must either be a BaggingRegressor or RandomForestClassifier
        n: The desired tree node
    """
    rf_estimators = model.estimators_.copy()
    features = np.array(X_train.columns)

    node_impurities, node_features = [], []

    for i, estimator in enumerate(rf_estimators):
        estimator_impurity =  # TODO 0
        estimator_feature =  # TODO 1

        node_impurities.append(estimator_impurity)
        node_features.append(estimator_feature)
    node_impurity_dict = {"feature":  # TODO
                              "impurity":  # TODO
    df =  # TODO
    return (df)
