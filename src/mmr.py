

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from src import content_based, most_popular

# MovieLens 1M's canonical genre vocabulary.
# We use this fixed vocabulary to ensure that every movie has the same
# feature columns, even if some genres are missing from the dataset.
GENRES = [
    "Action", "Adventure", "Animation", "Children's", "Comedy", "Crime",
    "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical",
    "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western",
]


def build_genre_vectors(movies):
    """Return binary MovieID-indexed genre vectors using the shared encoder."""
    required = {"MovieID", "Genres"}
    missing = required.difference(movies.columns)
    if missing:
        raise ValueError(f"movies is missing required columns: {sorted(missing)}")
    if movies["MovieID"].duplicated().any():
        raise ValueError("movies must contain one row per MovieID")
    encoded = content_based.encode_genres(movies)
    # Reindex to the complete MovieLens genre vocabulary, even if a genre is absent.
    return encoded.reindex(columns=GENRES, fill_value=0).astype(np.float32)


def mmr_rerank(candidates, genre_vectors, k=10, lambda_=0.5):
    """
    Re-rank one user's candidate movies using Maximal Marginal Relevance (MMR).

    MMR balances two objectives:
        1. Relevance: recommend movies with high original recommendation scores.
        2. Diversity: avoid selecting movies that are too similar to movies
           already selected for the recommendation list.

    The MMR formula is:

        MMR(i) = lambda * Relevance(i)
                 - (1 - lambda) * MaxSimilarity(i, SelectedMovies)

    Lambda controls the balance:
        lambda = 1.0: relevance only.
        lambda = 0.0: diversity only after the first selection.
        lambda = 0.5: equal weight for relevance and diversity.

    The first movie is selected using the highest relevance score.
    Each following movie is selected greedily using the MMR formula.

    Args:
        candidates (pd.DataFrame): Candidate movies with MovieID and Prediction.
        genre_vectors (pd.DataFrame): Binary genre features indexed by MovieID.
        k (int): Maximum number of movies to recommend.
        lambda_ (float): Weight controlling relevance versus diversity.

    Returns:
        pd.DataFrame: Re-ranked movies with their original scores, normalized
        relevance, MMR scores, and recommendation ranks.
    """

    # Check that k is a positive integer.
    if not isinstance(k, (int, np.integer)) or k <= 0:
        raise ValueError("k must be a positive integer")
    # Lambda must be a finite number between 0 and 1.
    if not np.isfinite(lambda_) or not 0.0 <= float(lambda_) <= 1.0:
        raise ValueError("lambda_ must be a finite number between 0 and 1")
    required = {"MovieID", "Prediction"}
    missing = required.difference(candidates.columns)
    if missing:
        raise ValueError(f"candidates is missing required columns: {sorted(missing)}")
    # Define the columns returned by this function.
    columns = ["MovieID", "Prediction", "BasePrediction", "Relevance", "MMRScore", "Rank"]
    if candidates.empty:
        return pd.DataFrame(columns=columns)
    if not genre_vectors.index.is_unique:
        raise ValueError("genre_vectors index (MovieID) must be unique")
    # Keep only MovieID and the original recommendation score.
    work = candidates[["MovieID", "Prediction"]].copy()
    # Convert prediction scores to numerical values.
    # Invalid or missing scores are removed.
    work["Prediction"] = pd.to_numeric(work["Prediction"], errors="coerce")
    work = work.dropna(subset=["MovieID", "Prediction"])
    work = work[np.isfinite(work["Prediction"].to_numpy(dtype=float))]
    work = work[work["MovieID"].isin(genre_vectors.index)]
    work = work.drop_duplicates("MovieID", keep="first")
    work = work.sort_values(["Prediction", "MovieID"], ascending=[False, True], kind="mergesort").reset_index(drop=True)
    # Return an empty result if no valid candidates remain.
    if work.empty:
        return pd.DataFrame(columns=columns)

    work["BasePrediction"] = work["Prediction"].astype(float)
    low = float(work["BasePrediction"].min())
    high = float(work["BasePrediction"].max())
    if high == low:
        work["Relevance"] = 1.0
    else:
        work["Relevance"] = (work["BasePrediction"] - low) / (high - low)

    movie_ids = work["MovieID"].tolist()
    vectors = genre_vectors.loc[movie_ids].to_numpy(dtype=np.float64, copy=True)
    # Store normalized relevance scores as a NumPy array.
    relevance = work["Relevance"].to_numpy(dtype=np.float64)

    # cosine_similarity defines zero-vector similarity as zero, which is suitable
    # for a movie with no genre metadata.
    # Keep track of:
    # selected  = candidate positions already added to the recommendation list.
    # remaining = candidate positions that have not been selected yet.
    # rows      = selected positions and their corresponding MMR scores.
    selected, remaining, rows = [], list(range(len(work))), []
    first = 0  # work is already sorted by relevance then MovieID.
    selected.append(first)
    remaining.remove(first)
    rows.append((first, float(relevance[first])))

    while len(selected) < min(k, len(work)) and remaining:
        similarities = cosine_similarity(vectors[remaining], vectors[selected])
        max_similarity = similarities.max(axis=1)
        scores = float(lambda_) * relevance[remaining] - (1.0 - float(lambda_)) * max_similarity
        # remaining retains original sorted order, so argmax resolves ties by MovieID.
        offset = int(np.argmax(scores))
        chosen = remaining.pop(offset)
        selected.append(chosen)
        rows.append((chosen, float(scores[offset])))

    out = work.iloc[[pos for pos, _ in rows]].copy().reset_index(drop=True)
    out["MMRScore"] = [score for _, score in rows]
    out["Rank"] = np.arange(1, len(out) + 1, dtype=int)
    # Evaluator sorts descending Prediction, then MovieID. Unique rank keys preserve
    # the MMR sequence regardless of the numerical magnitude/sign of MMRScore.
    out["Prediction"] = (len(out) - out["Rank"] + 1).astype(float)
    return out[columns]


def mmr_recommendations(predictions, genre_vectors, k=10, lambda_=0.5):
    """
    Apply MMR re-ranking independently to every user.

    Each user has a different candidate movie list.
    Therefore, the MMR selection process is performed separately for each user.

    Args:
        predictions (pd.DataFrame): Candidate recommendations containing
            UserID, MovieID, and Prediction.
        genre_vectors (pd.DataFrame): Binary genre features indexed by MovieID.
        k (int): Maximum number of recommendations per user.
        lambda_ (float): Relevance-diversity balance parameter.

    Returns:
        pd.DataFrame: MMR-ranked recommendations for all users.
    """
    output_columns = ["UserID", "MovieID", "Prediction", "BasePrediction", "Relevance", "MMRScore", "Rank"]
    required = {"UserID", "MovieID", "Prediction"}
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(f"predictions is missing required columns: {sorted(missing)}")
    if predictions.empty:
        return pd.DataFrame(columns=output_columns)
    pieces = []
    for user_id, group in predictions.groupby("UserID", sort=True):
        ranked = mmr_rerank(group[["MovieID", "Prediction"]], genre_vectors, k=k, lambda_=lambda_)
        if not ranked.empty:
            ranked.insert(0, "UserID", user_id)
            pieces.append(ranked)
    return pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame(columns=output_columns)


def generate_mmr_recommendations(
    target_users, train, catalogue_ids, movies, k=10, lambda_=0.5,
    window_days=None, min_rating=4,
):
    """
    Generate final movie recommendations using Most Popular + MMR.

    This function connects the existing Most Popular recommender with the
    MMR re-ranking algorithm.

    Process:
        1. Calculate movie popularity using the selected settings.
        2. Generate candidate recommendations for target users.
        3. Exclude movies that users have already rated.
        4. Convert movie genres into binary numerical features.
        5. Apply MMR independently to each user's candidate movies.
        6. Return the final ranked recommendations.

    Args:
        target_users: Users for whom recommendations should be generated.
        train (pd.DataFrame): Rating history used to calculate popularity.
        catalogue_ids: MovieIDs belonging to the movie catalogue.
        movies (pd.DataFrame): Movie information including genres.
        k (int): Maximum number of recommendations per user.
        lambda_ (float): Weight controlling relevance versus diversity.
        window_days (int or None): Optional popularity time window.
        min_rating (int): Minimum rating counted when calculating popularity.

    Returns:
        pd.DataFrame: Final MMR recommendations for all target users.
    """
    if not np.isfinite(lambda_) or not 0.0 <= float(lambda_) <= 1.0:
        raise ValueError("lambda_ must be a finite number between 0 and 1")
    popularity = most_popular.compute_popularity(
        train=train, catalogue_ids=catalogue_ids,
        window_days=window_days, min_rating=min_rating,
    )
    predictions = most_popular.generate_prediction_scores(
        target_users=target_users, train=train, popularity=popularity,
    )
    return mmr_recommendations(
        predictions=predictions, genre_vectors=build_genre_vectors(movies),
        k=k, lambda_=lambda_,
    )
