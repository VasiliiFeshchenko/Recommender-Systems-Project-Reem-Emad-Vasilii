import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from surprise import Dataset, Reader
from src import (
    item_based_kNN,
    most_popular,
    user_based_kNN,
    matrix_factorization,
    content_based,
    services,
)
from src.data_loader import load_ratings, load_movies
from src.item_based_kNN import get_target_users
from src.evaluation import evaluate_ranking


def sample_training_data(train, fraction, seed=42):

    sampled_train = train.sample(
        frac=fraction,
        random_state=seed
    )

    return sampled_train.sort_index()

def prepare_ablation_data(data_dir):

    ratings = load_ratings(data_dir)
    movies = load_movies(data_dir)

    # Sort ratings chronologically, just like run_all.py
    order = np.argsort(ratings["Timestamp"].to_numpy(), kind="stable")
    ratings_sorted = ratings.iloc[order].reset_index(drop=True)

    # Separate the final 20% as test data
    test_start = int(len(ratings_sorted) * 0.80)

    history = ratings_sorted.iloc[:test_start].copy()
    test = ratings_sorted.iloc[test_start:].copy()

    # Select eligible users using the full training history
    target_users = get_target_users(test, history)

    return history, test, movies, target_users


    

def prepare_training_samples(history, seed=42):

    fractions = [1.0, 0.5, 0.2]

    samples = {}

    for fraction in fractions:
        samples[fraction] = sample_training_data(
            history, fraction, seed
        )

    # Verify that 20% is contained within 50%
    assert samples[0.2].index.isin(samples[0.5].index).all()

    # Verify that 50% is contained within 100%
    assert samples[0.5].index.isin(samples[1.0].index).all()

    return samples

def get_common_target_users(samples, test):

    # Users with at least 5 ratings in the smallest training sample
    eligible_users = get_target_users(test, samples[0.2])

    # Relevant test movies for each user
    relevant = test[test["Rating"] >= 4]
    relevant_movies = relevant.groupby("UserID")["MovieID"].agg(set).to_dict()

    # Movies already rated in the full training history
    seen_movies = (
        samples[1.0].groupby("UserID")["MovieID"].agg(set).to_dict()
    )

    # Keep users with at least one relevant unseen test movie
    common_users = [
        user for user in eligible_users
        if relevant_movies.get(user, set()) - seen_movies.get(user, set())
    ]

    return np.array(common_users)

def evaluate_item_knn_sample(
    sampled_train, full_history, test, movies, common_users, settings
):

    # Build the movie-user matrix using the sampled training data
    matrix = item_based_kNN.build_movie_user_matrix(sampled_train)

    # Select the similarity method from frozen settings
    similarity_methods = {
        "Cosine": item_based_kNN.compute_cosine_similarity,
        "Pearson": item_based_kNN.compute_pearson_similarity,
        "Jaccard": item_based_kNN.compute_jaccard_similarity
    }

    compute_similarity = similarity_methods[settings["similarity"]]
    similarities = compute_similarity(matrix)

    # Generate predictions using the selected k
    predictions = item_based_kNN.generate_prediction_scores(
        common_users,
        sampled_train,
        matrix,
        similarities,
        k=int(settings["k"]),
        catalogue_ids=movies["MovieID"],
        verbose=False
    )

    # Keep the candidate exclusions fixed across training fractions
    seen_pairs = pd.MultiIndex.from_frame(
        full_history[["UserID", "MovieID"]]
    )

    prediction_pairs = pd.MultiIndex.from_frame(
        predictions[["UserID", "MovieID"]]
    )

    predictions = predictions.loc[
        ~prediction_pairs.isin(seen_pairs)
    ]

    # Evaluate using the common users and the original test set
    metrics = evaluate_ranking(
        predictions,
        test,
        full_history,
        common_users,
        movies["MovieID"],
        cutoff=10,
        relevance_threshold=4
    )

    return metrics

def evaluate_popularity_sample(
    sampled_train, full_history, test, movies, common_users, settings
):

    # Calculate popularity using the sampled training data
    popularity = most_popular.compute_popularity(
        sampled_train,
        movies["MovieID"],
        window_days=settings["window_days"],
        min_rating=settings["min_rating"]
    )

    # Generate recommendations for the common users
    predictions = most_popular.generate_prediction_scores(
        common_users,
        sampled_train,
        popularity
    )

    # Exclude movies already rated in the full training history
    seen_pairs = pd.MultiIndex.from_frame(
        full_history[["UserID", "MovieID"]]
    )

    prediction_pairs = pd.MultiIndex.from_frame(
        predictions[["UserID", "MovieID"]]
    )

    predictions = predictions.loc[
        ~prediction_pairs.isin(seen_pairs)
    ]

    # Evaluate using the same test set and users
    metrics = evaluate_ranking(
        predictions,
        test,
        full_history,
        common_users,
        movies["MovieID"],
        cutoff=10,
        relevance_threshold=4
    )

    return metrics

def evaluate_user_knn_sample(
    sampled_train, full_history, test, movies, common_users, settings
):

    matrix = user_based_kNN.build_user_movie_matrix(sampled_train)

    similarity_methods = {
        "Cosine": user_based_kNN.compute_cosine_similarity,
        "Pearson": user_based_kNN.compute_pearson_similarity,
        "Jaccard": user_based_kNN.compute_jaccard_similarity
    }

    similarities = similarity_methods[settings["similarity"]](matrix)

    catalogue = pd.Index(movies["MovieID"]).unique()
    user_means = sampled_train.groupby("UserID")["Rating"].mean()
    movie_means = sampled_train.groupby("MovieID")["Rating"].mean()
    global_mean = sampled_train["Rating"].mean()

    seen_full = full_history.groupby("UserID")["MovieID"].agg(set).to_dict()

    predictions = []

    for user in common_users:

        if user not in matrix.index:
            raise ValueError(f"User {user} missing from training sample")

        # Choose the closest positive-similarity neighbours
        neighbors = similarities.loc[user].drop(index=user)
        neighbors = neighbors[neighbors > 0].nlargest(int(settings["k"]))

        # Default scores for movies without neighbour information
        scores = movie_means.reindex(catalogue).fillna(global_mean).copy()

        if not neighbors.empty:

            neighbor_ratings = matrix.loc[neighbors.index].to_numpy(dtype=float)
            observed = np.isfinite(neighbor_ratings)

            weights = neighbors.to_numpy()[:, None] * observed

            neighbor_means = user_means.reindex(
                neighbors.index
            ).to_numpy()[:, None]

            deviations = np.where(
                observed,
                neighbor_ratings - neighbor_means,
                0.0
            )

            denominator = weights.sum(axis=0)
            numerator = (weights * deviations).sum(axis=0)

            valid = np.flatnonzero(denominator > 0)

            predicted = (
                user_means.loc[user]
                + numerator[valid] / denominator[valid]
            )

            scores.loc[matrix.columns[valid]] = predicted

        # Remove full-history movies before ranking
        scores = scores.drop(
            index=list(seen_full.get(user, set())),
            errors="ignore"
        )

        ranked = pd.DataFrame({
            "MovieID": scores.index,
            "Prediction": scores.to_numpy()
        })

        ranked = ranked.sort_values(
            ["Prediction", "MovieID"],
            ascending=[False, True]
        ).head(10)

        ranked["UserID"] = user
        predictions.append(ranked)

    predictions = pd.concat(predictions, ignore_index=True)

    return evaluate_ranking(
        predictions, test, full_history, common_users,
        movies["MovieID"], cutoff=10, relevance_threshold=4
    )

def evaluate_mf_sample(
    sampled_train, full_history, test, movies, common_users, settings
):

    data = Dataset.load_from_df(
        sampled_train[["UserID", "MovieID", "Rating"]],
        Reader(rating_scale=(1, 5))
    )

    trainset = data.build_full_trainset()

    matrix = services.build_movie_user_matrix(sampled_train)

    movie_ids = matrix.index.to_numpy()

    unrated_mask = (
        matrix.loc[:, common_users]
        .isna()
        .T
        .to_numpy()
    )

    cold_movies = pd.Index(movies["MovieID"]).unique().difference(
        matrix.index, sort=False
    )

    predictions = matrix_factorization.get_predictions(
        trainset,
        common_users,
        movie_ids,
        unrated_mask,
        cold_movies,
        sampled_train["Rating"].mean(),
        **settings
    )

    # Exclude movies rated in the full history
    seen_pairs = pd.MultiIndex.from_frame(
        full_history[["UserID", "MovieID"]]
    )

    prediction_pairs = pd.MultiIndex.from_frame(
        predictions[["UserID", "MovieID"]]
    )

    predictions = predictions.loc[
        ~prediction_pairs.isin(seen_pairs)
    ]

    return evaluate_ranking(
        predictions, test, full_history, common_users,
        movies["MovieID"], cutoff=10, relevance_threshold=4
    )

def evaluate_content_sample(
    sampled_train, full_history, test, movies, common_users, settings
):

    features = content_based.build_item_features(
        movies,
        year_weight=settings["year_weight"]
    )

    profiles = content_based.build_user_profiles(
        sampled_train,
        features,
        weight_mode=settings["weight_mode"]
    )

    seen_full = (
        full_history.groupby("UserID")["MovieID"].agg(set).to_dict()
    )

    batches = []

    for start in range(0, len(common_users), 50):

        batch_users = common_users[start:start + 50]

        predictions = content_based.generate_prediction_scores(
            batch_users, sampled_train, features, profiles
        )

        # Remove full-history interactions before selecting Top 10
        seen_pairs = {
            (user, movie)
            for user in batch_users
            for movie in seen_full.get(user, set())
        }

        keep = [
            (user, movie) not in seen_pairs
            for user, movie in zip(
                predictions["UserID"], predictions["MovieID"]
            )
        ]

        predictions = predictions.loc[keep]

        predictions = (
            predictions.sort_values(
                ["UserID", "Prediction", "MovieID"],
                ascending=[True, False, True]
            )
            .groupby("UserID")
            .head(10)
        )

        batches.append(predictions)

    all_predictions = pd.concat(batches, ignore_index=True)

    return evaluate_ranking(
        all_predictions, test, full_history, common_users,
        movies["MovieID"], cutoff=10, relevance_threshold=4
    )


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/ml-1m")
    )
    args = parser.parse_args()

    # Prepare dataset and samples once
    history, test, movies, target_users = prepare_ablation_data(
        args.data_dir
    )

    samples = prepare_training_samples(history, seed=42)
    common_users = get_common_target_users(samples, test)

    print("Common evaluable users:", len(common_users))

    # Load validation-selected hyperparameters
    with open("outputs/frozen_test_config.json", "r") as file:
        config = json.load(file)

    popularity_settings = next(
        variant
        for variant in config["popularity_variants"]
        if variant["name"] == config["preferred_popularity"]
    )

    # All five models use the same training samples
    models = [
        ("Item-Based kNN", evaluate_item_knn_sample, config["knn"]),
        ("Most Popular", evaluate_popularity_sample, popularity_settings),
        ("User-Based kNN", evaluate_user_knn_sample, config["user_based_knn"]),
        ("Matrix Factorization", evaluate_mf_sample, config["mf"]),
        ("Content-Based", evaluate_content_sample, config["content_based"])
    ]

    results = []

    for model_name, evaluate_function, settings in models:

        for fraction, sampled_data in samples.items():

            print(
                f"\nEvaluating {model_name} "
                f"with {fraction:.0%} training data..."
            )

            metrics = evaluate_function(
                sampled_data,
                history,
                test,
                movies,
                common_users,
                settings
            )

            results.append({
                "Model": model_name,
                "Training fraction": fraction,
                "Training interactions": len(sampled_data),
                "Evaluable users": len(common_users),
                "Seed": 42,
                **metrics
            })

            print(f"NDCG@10: {metrics['NDCG@10']:.6f}")

    # Build final comparison table
    comparison = pd.DataFrame(results)

    assert len(comparison) == 15
    assert comparison["NDCG@10"].notna().all()

    output_path = Path("outputs/sparsity_ablation_results.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    comparison.to_csv(output_path, index=False)

    print("\nFINAL SPARSITY ABLATION RESULTS")
    print(
        comparison.pivot(
            index="Model",
            columns="Training fraction",
            values="NDCG@10"
        ).round(6).to_string()
    )

    print("\nSaved:", output_path)


if __name__ == "__main__":
    main()




