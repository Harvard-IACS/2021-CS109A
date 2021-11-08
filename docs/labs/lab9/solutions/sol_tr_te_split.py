################################### Train Test split
np.random.seed(40)

#test_proportion
test_prop = 0.2
msk = np.random.uniform(0, 1, len(cancer_scaled)) > test_prop

#Split predictor and response columns
x_train, y_train =  cancer_scaled[msk], target[msk]
x_test , y_test  = cancer_scaled[~msk], target[~msk]

print("Shape of Training Set :", x_train.shape)
print("Shape of Testing Set :" , x_test.shape)