# Import the necessary libraries
import warnings

import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings("ignore")


# Helper function to plot the impurity-based feature importances of the defined model
def plot_feature_importance(model1, model2, X, y):
    plt.xkcd(scale=0.3, randomness=0.6)
    #     maxlim = max(max(model1.feature_importances_),max(model2.feature_importances_))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Plot for Random Forest
    tree_importance_sorted_idx = np.argsort(model1.feature_importances_)
    tree_indices = np.arange(0, len(model1.feature_importances_)) + 0.5

    ax1.barh(tree_indices,
             model1.feature_importances_[tree_importance_sorted_idx],
             height=0.7, color='#B2D7D0')
    ax1.set_yticks(tree_indices)
    ax1.set_yticklabels(X.columns[tree_importance_sorted_idx],
                        fontsize=12)
    ax1.set_ylim((0, len(model1.feature_importances_)))
    #     ax1.set_xlim((0, maxlim+0.01))
    ax1.set_xlabel("Impurity Based Feature Importance", fontsize=16)

    # ax1.set_ylabel("Predictors", fontsize=16)

    # Plot for Bagging

    tree_importance_sorted_idx = np.argsort(model2.feature_importances_)
    tree_indices = np.arange(0, len(model2.feature_importances_)) + 0.5
    difference = model2.feature_importances_ - model1.feature_importances_
    difference = difference[tree_importance_sorted_idx]

    ax2.barh(tree_indices, model2.feature_importances_[tree_importance_sorted_idx],
             height=0.7, color='#EFAEA4')
    for index, value in enumerate(model2.feature_importances_[tree_importance_sorted_idx]):
        ax2.text(value, index + 0.3, f" {str(round(difference[index], 3))}", fontsize=14)

    ax2.set_yticks(tree_indices)
    ax2.set_yticklabels(X.columns[tree_importance_sorted_idx], fontsize=12)

    ax2.set_ylim((0, len(model2.feature_importances_)))
    maxlim = max(model2.feature_importances_)
    ax2.set_xlim(0, maxlim + 0.02)
    ax2.set_xlabel("Impurity Based Feature Importance", fontsize=16)

    ax1.set_title("Single Tree", fontsize=18)
    ax2.set_title("Random Forest", fontsize=18)
    fig.tight_layout()
    plt.show()


# Helper function to plot the feature importance for the defined model
def plot_permute_importance(result1, result2, X, y):
    #     maxlim = max(max(result1.importances_mean),max(result2.importances_mean))
    plt.xkcd(scale=0.3, randomness=0.6)
    # Plot for random forest
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    tree_importance_sorted_idx = np.argsort(result1.importances_mean)
    tree_indices = np.arange(0, len(result1.importances_mean)) + 0.5

    ax1.barh(tree_indices, result1.importances_mean[tree_importance_sorted_idx],
             height=0.7, color='#B2D7D0')
    ax1.set_yticks(tree_indices)
    ax1.set_yticklabels(X.columns[tree_importance_sorted_idx], fontsize=12)
    ax1.set_ylim((0, len(result1.importances_mean)))
    ax1.set_xlabel("Permutation Feature Importance", fontsize=16)

    # ax1.set_ylabel("Predictors", fontsize=16)

    # Plot for Bagging

    tree_importance_sorted_idx2 = np.argsort(result2.importances_mean)
    tree_indices2 = np.arange(0, len(result2.importances_mean)) + 0.5
    difference = result2['importances_mean'] - result1['importances_mean']
    difference = difference[tree_importance_sorted_idx]

    ax2.barh(tree_indices2, result2.importances_mean[tree_importance_sorted_idx2],
             height=0.7, color='#EFAEA4')
    for index, value in enumerate(result2.importances_mean[tree_importance_sorted_idx2]):
        ax2.text(value, index + 0.3, f" {str(round(difference[index], 3))}", fontsize=14)

    ax2.set_yticks(tree_indices2)
    ax2.set_yticklabels(X.columns[tree_importance_sorted_idx2], fontsize=12)
    ax2.set_ylim((0, len(result2.importances_mean)))
    ax2.set_xlabel("Permutation Feature Importance", fontsize=16)
    maxlim = max(result2.importances_mean)
    ax2.set_xlim(0, maxlim + 0.015)
    ax1.set_title("Single Tree", fontsize=18)
    ax2.set_title("Random Forest", fontsize=18)
    fig.tight_layout()
    plt.show()
