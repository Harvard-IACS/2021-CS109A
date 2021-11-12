tree_depth_range = range(1, 20, 2)
rf_val_acc = get_tree_scores(ex1_x_train, 
                             ex1_y_train, 
                             RandomForestClassifier(), 
                             tree_depth_range)
#rf_mean_acc = pd.DataFrame(rf_val_acc)
rf_mean_acc  = rf_val_acc.groupby("depth").mean()
rf_mean_acc["depth"] = list(tree_depth_range)
rf_mean_acc