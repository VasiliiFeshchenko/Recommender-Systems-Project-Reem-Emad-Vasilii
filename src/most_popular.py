import pandas as pd

POPULARITY_WINDOWS = [("All training history", None), ("Last 90 days", 90), ("Last 30 days", 30)]
RATING_THRESHOLDS = [3, 4, 5]


def compute_popularity(train, catalogue_ids, window_days=None, min_rating=4):
    """Count qualifying training ratings for every movie in the catalogue."""
    ratings_to_count = train
    if window_days is not None:
        window_end = train["Timestamp"].max()
        window_start = window_end - window_days * 24 * 60 * 60
        ratings_to_count = train[train["Timestamp"] >= window_start]

    ratings_to_count = ratings_to_count[ratings_to_count["Rating"] >= min_rating]
    counts = ratings_to_count.groupby("MovieID").size()
    # Keep cold movies and movies without qualifying ratings as zero-score candidates.
    catalogue = pd.Index(catalogue_ids).unique()
    return counts.reindex(catalogue, fill_value=0)


def generate_prediction_scores(target_users, train, popularity):
    """Use global popularity scores, excluding each user's full training history."""
    seen_movies = train.groupby("UserID")["MovieID"].agg(set).to_dict()
    predictions = []
    for user in target_users:
        # Exclude all training-rated movies, including low ratings and older ones.
        candidates = popularity[~popularity.index.isin(seen_movies.get(user, set()))]
        predictions.append(pd.DataFrame({
            "UserID": user,
            "MovieID": candidates.index,
            "Prediction": candidates.to_numpy(),
        }))

    if not predictions:
        return pd.DataFrame(columns=["UserID", "MovieID", "Prediction"])
    return pd.concat(predictions, ignore_index=True)
