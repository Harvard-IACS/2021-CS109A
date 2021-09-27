import numpy
from sklearn.preprocessing import PolynomialFeatures


# This a helper function that will help you plot the estimated polynomial regression along with the true values & true function
# We will use this later

def plot_functions(d, est, ax, df, alpha, xvalid, yvalid, xtrain, ytrain):
    """Plot the approximation of ``est`` on axis ``ax``. """
    ax.plot(df.x, df.f, color='k', label='f')
    ax.plot(xtrain, ytrain, 's', label="training", ms=5, alpha=0.7, color='darkblue')
    ax.plot(xvalid, yvalid, 's', label="validation", ms=5, alpha=0.8, color='#007D66')
    transx = numpy.arange(0, 1.1, 0.01)
    transX = PolynomialFeatures(d).fit_transform(transx.reshape(-1, 1))
    ax.plot(transx, est.predict(transX), linewidth=3, alpha=0.8, label="alpha = %s" % str(alpha), color='#FF2F92')

    # This is just aesthetics
    ax.set_ylim((1, 2))
    ax.set_xlim((0, 1))
    ax.set_ylabel('y')
    ax.set_xlabel('x')
    ax.legend(loc='lower right')


# This helper function will help you plot the magnitude of the coefficients of the polynomial regression
# The values will help you determine which powers of the feature are most influential

def plot_coefficients(est, ax, alpha):
    coef = est.coef_.ravel()
    ax.bar(range(len(coef)), numpy.abs(coef), label=f'alpha = {alpha}', color='#9FC131FF', alpha=0.5, edgecolor='k')
    ax.set_yscale('log')
    ax.set_ylim((1e-1, 1e15))
    ax.set_ylabel('abs(coefficient)')
    ax.set_xlabel('coefficients')
    ax.legend(loc='upper left')
