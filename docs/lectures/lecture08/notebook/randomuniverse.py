def RandomUniverse(df):
    df_bootstrap = df.sample(200, replace=True)
    return df_bootstrap
