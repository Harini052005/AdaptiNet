def build_hypergraph_features(df):
    grouped = df.groupby(["post", "hashtag", "time"])["engagement"].sum().reset_index()
    return grouped