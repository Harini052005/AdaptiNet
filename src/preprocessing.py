def clean_data(df):
    df["engagement"] = df["likes"] + 2 * df["shares"]

    threshold = df["engagement"].mean() + 2 * df["engagement"].std()
    df_clean = df[df["engagement"] < threshold]

    return df_clean