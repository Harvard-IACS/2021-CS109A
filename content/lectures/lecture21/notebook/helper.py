import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import ListedColormap

sns.set_style('white')
import pandas as pd

df = pd.read_csv("boostingclassifier.csv")
X = df[['latitude', 'longitude']].values

y = df['landtype'].values


def plot_decision_boundary(classifier, X, y, N=10, scatter_weights=np.ones(len(y)), ax=None, counter=0, label=False):
    '''Utility function to plot decision boundary and scatter plot of data'''
    x_min, x_max = X[:, 0].min() - .1, X[:, 0].max() + .1
    y_min, y_max = X[:, 1].min() - .1, X[:, 1].max() + .1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, N), np.linspace(y_min, y_max, N))
    cmap = ListedColormap(["#ABCCE3", "#50AEA4"])

    # Check what methods are available
    if hasattr(classifier, "decision_function"):
        zz = np.array([classifier.decision_function(np.array([xi, yi]).reshape(1, -1)) for xi, yi in
                       zip(np.ravel(xx), np.ravel(yy))])
    elif hasattr(classifier, "predict_proba"):
        zz = np.array([classifier.predict_proba(np.array([xi, yi]).reshape(1, -1))[:, 1] for xi, yi in
                       zip(np.ravel(xx), np.ravel(yy))])
    else:
        zz = np.array([classifier(np.array([xi, yi]).reshape(1, -1)) for xi, yi in zip(np.ravel(xx), np.ravel(yy))])

    # reshape result and plot
    Z = zz.reshape(xx.shape)
    cm_bright = ListedColormap(["#EFAEA4", "#F6345E"])

    # Get current axis and plot
    if ax is None:
        ax = plt.gca()
    ax.contourf(xx, yy, Z, cmap=cmap, alpha=0.4)
    if label:
        ax.scatter(X[:, 0], X[:, 1], c=y, cmap=cm_bright, s=scatter_weights * 40, edgecolors='k',
                   label=f'Stump {counter}')
        ax.legend(fontsize=16)
    else:
        ax.scatter(X[:, 0], X[:, 1], c=y, cmap=cm_bright, s=scatter_weights * 40, edgecolors='k',
                   label=f'Stump {counter}')
    ax.set_xlabel('$Latitude$', fontsize=14)
    ax.set_ylabel('$Longitude$', fontsize=14)
    ax.set_title(f'Stump {counter + 1} decision boundary', fontsize=16)
