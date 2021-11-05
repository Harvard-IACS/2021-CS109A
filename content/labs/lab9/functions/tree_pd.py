import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.metrics as metrics
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample

def get_tree_scores(x_train, y_train, model, tree_depth_range, bootstraps = 1):
    """
    This function will take a model and a range of values and return a dataframe with columns: ["depth", "cross_val_acc"]
    
    Arguments:
        model: the model to be run. Specifically this should be a class instance such as
            DecisionTreeClassifier().
        tree_depth_range: the range of values over which the tree depths should be saved.
        
    """
    x_train_orig = x_train.copy()
    y_train_orig = y_train.copy()

    mean_CV_acc = {}
    all_CV_acc = {}

    #declare dictionaries
    for boot in range(bootstraps):
        
        
        tree_depth_list = list(tree_depth_range)
        
        #put tree depths into a list
        tree_depths = list(tree_depth_range)
        
        #find and store cross_validated scores
        for depth in tree_depths:
            
            model.max_depth = depth

            if bootstraps > 1:
                x_train_boot, y_train_boot = resample(x_train, y_train)
                score = cross_val_score(estimator=model, X=x_train_boot, y=y_train_boot, n_jobs=-1, cv= 10)
            else:
                score = cross_val_score(estimator=model, X=x_train, y=y_train, n_jobs=-1, cv= 10)
            if all_CV_acc.get(depth):
                [all_CV_acc[depth].append(score_) for score_ in list(score)]
            else:
                all_CV_acc[depth] = list(score) 
            mean_CV_acc[depth] = score.mean() 
    
    #make the dataframes
    # cv_acc_pd = pd.melt(pd.DataFrame(all_CV_acc))
    # cv_acc_pd.columns = ["depth", "cv_acc_score"]
    return all_CV_acc #cv_acc_pd     

def load_cancer_dataset(n, random_state):
    
    random_state = np.random.RandomState(random_state)

    #load the data
    dataset = datasets.load_breast_cancer() 
    num_features = min(n, dataset.data.shape[1]) 


    features_to_keep = random_state.choice(dataset.data.shape[1],
                                           size=num_features, 
                                           replace = False)

    cancer = dataset.data[:, features_to_keep ] 

    cancer_scaled = StandardScaler().fit_transform(cancer)

    cancer_scaled = pd.DataFrame(cancer_scaled)
    
    cancer_scaled.columns = dataset.feature_names[features_to_keep]

    print("Design matrix shape {:}".format(cancer_scaled.shape))
    display(cancer_scaled.head(4))

    ################################### Assign the target
    target = dataset.target.reshape(-1,1)
    target_dict = {1 : "Benign", 0: "Malignant"}
    print('target classes   ', target_dict)
    n_benign, n_malignant = sum(dataset.target == 1), sum(dataset.target != 1)
    print("There are {:} Benign cases and {:} Malignant cases in the target".format(n_benign, n_malignant))


    
    return cancer_scaled, target