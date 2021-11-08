# init exercise 1 solution

# Train model
model1_wine = LogisticRegression(penalty='none').fit(X_wine_train, y_wine_train)

# Score model
train_score = model1_wine.score(X_wine_train, y_wine_train)
test_score = model1_wine.score(X_wine_test, y_wine_test)

# Print scores
print("Training Set Accuracy: {:.4f}".format(train_score))
print("Testing Set Accuracy: {:.4f}\n".format(test_score))

# Predict probabilities for our training data
y_proba_train = model1_wine.predict_proba(X_wine_train)

# Check shape of our predictions to show that we have 3 probabilities predicted
# for each observation (i.e. predicted probabilities for each of our 3 classes)
print(
    "The shape of our predicted training probabilities array: {}\n"
    .format(y_proba_train.shape)
)

# Sum all 3 classes at each observation
print(
    "The sum of predicted probabilities for all 3 classes by observation:\n\n{}"
    .format(np.sum(y_proba_train, axis=1))
)
