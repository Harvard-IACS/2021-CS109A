#hint use the fit and transform calls.
scaler = StandardScaler().fit(train)
train_standard = scaler.transform(train)
val_standard = scaler.transform(val)
test_standard = scaler.transform(test)