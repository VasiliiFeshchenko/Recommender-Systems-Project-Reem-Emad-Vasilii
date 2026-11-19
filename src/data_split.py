def chronological_split(ratings):
    ratings_sorted = ratings.sort_values(
        "Timestamp", kind="stable"
    ).reset_index(drop=True)

    n = len(ratings_sorted)
    train_end = int(n * 0.70)
    validation_end = int(n * 0.80)

    train = ratings_sorted.iloc[:train_end].copy()
    validation = ratings_sorted.iloc[train_end:validation_end].copy()
    test = ratings_sorted.iloc[validation_end:].copy()

    return train, validation, test