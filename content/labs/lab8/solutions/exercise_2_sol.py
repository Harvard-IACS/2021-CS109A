# init exercise 2 solution

# Using an approach similar to what was used in the Iris example
# we can identify appropriate boundaries for our meshgrid by
# referencing the actual wine data

x_1_wine = X_wine_train[predictors[0]]
x_2_wine = X_wine_train[predictors[1]]

x_1_min_wine, x_1_max_wine = x_1_wine.min() - 0.2, x_1_wine.max() + 0.2
x_2_min_wine, x_2_max_wine = x_2_wine.min() - 0.2, x_2_wine.max() + 0.2

# Then we use np.arange to generate our interval arrays
# and np.meshgrid to generate our actual grids

xx_1_wine, xx_2_wine = np.meshgrid(
    np.arange(x_1_min_wine, x_1_max_wine, 0.003),
    np.arange(x_2_min_wine, x_2_max_wine, 0.003)
)

# Now we have everything we need to generate our plot

plot_wine_2d_boundaries(
    X_wine_train,
    y_wine_train,
    predictors,
    model1_wine,
    xx_1_wine,
    xx_2_wine,
)