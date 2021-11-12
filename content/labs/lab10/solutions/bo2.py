from tqdm.notebook import trange

#logarithmic values:
exp_powers = list(range(-6,1))
exp_vals = list(np.exp(exp_powers))

#Find Optimal Learning Rate for Ada-Boosting
staged_train_scores, staged_test_scores = {}, {}
score_train, score_test = {}, {}
for i in trange(len(exp_vals)):
    model = AdaBoostClassifier(
        base_estimator=DecisionTreeClassifier(max_depth=3),
        n_estimators=200, learning_rate=exp_vals[i])
    model.fit(x_train.values, y_train)
    score_train[exp_vals[i]] = accuracy_score(y_train, model.predict(x_train.values))
    score_test[exp_vals[i]] = accuracy_score(y_test, model.predict(x_test.values))
    
    staged_train_scores[exp_vals[i]] = list(model.staged_score(x_train.values, y_train))
    staged_test_scores[exp_vals[i]] = list(model.staged_score(x_test.values,  y_test))
    
#Plot
lists1 = sorted(score_train.items())
lists2 = sorted(score_test.items())
x1, y1 = zip(*lists1) 
x2, y2 = zip(*lists2) 
plt.figure(figsize=(10,7))
plt.ylabel("Accuracy")
plt.xlabel("Log Learning Rate Log($\lambda$)")
plt.title('Variation of Accuracy with Depth - ADA Boost Classifier')
plt.plot(np.log(x1), y1, 'b-', label='Train')
plt.plot(np.log(x2), y2, 'g-', label='Test')
plt.legend()
plt.show()

##export this to a function or delete it.
fig, ax = plt.subplots(1,2, figsize=(10,7))
for key, val in staged_train_scores.items():
    ax[0].plot(list(val),label='train')
    
for i, (key, val) in enumerate(staged_test_scores.items()):
    ax[1].plot(list(val),label='$\lambda$=exp({})'.format(exp_powers[i]))
    ax[1].set_title("h")
plt.legend(loc = 4)

sets = ["Train", "Test"]
for i in range(2):
    ax[i].set_xlabel('Iteration')
    ax[i].set_ylabel('Accuracy')
    ax[i].set_title(sets[i] + " Set Accuracy vs Iterations - ADA Boost")

plt.show()

