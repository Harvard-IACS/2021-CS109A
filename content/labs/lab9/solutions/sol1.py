#1
tree_depth_range = range(1, 15, 1)
tree_Scores = get_tree_scores(x_train, 
                        y_train, 
                        DecisionTreeClassifier(random_state = 42), 
                        tree_depth_range)

#2
cv_acc_df = pd.melt(pd.DataFrame(tree_Scores), var_name = "depth", value_name = 'cv_acc_score')

#3
plt.figure(figsize=(12, 3))
plt.title('Variation of Accuracy on Validation set with Depth - Simple Decision Tree')
sns.boxenplot(x = "depth", y = "cv_acc_score", data = cv_acc_df);
plt.show()

#4
cv_acc_mean = cv_acc_df.groupby("depth").mean()
cv_acc_mean["depth"] = list(tree_depth_range)

#5
plt.figure(figsize=(12, 3))
plt.title('Mean Validation set accuracy score — simple decision tree')
sns.lineplot(x = "depth", y = "cv_acc_score", data = cv_acc_mean);