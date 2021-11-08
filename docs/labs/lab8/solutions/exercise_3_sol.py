# init exercise 3 solution

# create interaction term for both train and test sets

for df in [X_wine_train, X_wine_test]:
    df["_".join(predictors)] = df[predictors[0]] * df[predictors[1]]

# View the resulting dataframe
display(X_wine_train.head())

print()

# Fit cv logistic regression model to predictors (including interaction term)

Cs = [1e-4,1e-3,1e-2,1e-1,1e0,1e1,1e2,1e3,1e4]
cv = 3
penalty = 'l1'
solver = 'liblinear'

model2_wine = LogisticRegressionCV(
    Cs=Cs, cv=cv, penalty=penalty, solver='liblinear'
).fit(X_wine_train, y_wine_train)

model2_score_train = model2_wine.score(X_wine_train, y_wine_train)
model2_score_test = model2_wine.score(X_wine_test, y_wine_test)

model2_coefficients = np.hstack(
    [model2_wine.intercept_.reshape(-1, 1), model2_wine.coef_]
).T

print(
    "The regularization parameter C chosen by this model for each class "
    "was:\n\n\t{}\n\n"
    "The accuracy scores for this model are:"
    "\n\n\tTrain\t{:.4f}\n\tTEST\t{:.4f}\n"
    .format(model2_wine.C_, model2_score_train, model2_score_test)
)

print(
    "The coefficients for this model by class are:"
    "\n\n\t\t\t\tclass\n\t\t\t\t0\t\t1\t\t2\n"
)

coef_names = ["intercept"] + list(X_wine_train.columns)

for name, values in zip(coef_names, model2_coefficients):
    coefs_formatted = ["{:.4f}".format(val) for val in values]
    print("\t{}   \t\t{}".format(name, "\t\t".join(coefs_formatted)))