
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def top_ten(predictions, k):
    """Keep the top k per user with the team's MovieID tie-breaking."""
    return (
        predictions.sort_values(
            ["UserID", "Prediction", "MovieID"],
            ascending=[True, False, True],
        )
        .groupby("UserID", sort=False)
        .head(k)
        .reset_index(drop=True)
    )


def evaluable_users(held_out, history, target_users, threshold):
    """Match src.evaluation: a user needs >=1 relevant unseen held-out movie."""
    seen = history.groupby("UserID")["MovieID"].agg(set).to_dict()
    relevant = (
        held_out.loc[held_out["Rating"] >= threshold]
        .groupby("UserID")["MovieID"].agg(set).to_dict()
    )
    return [
        int(u) for u in target_users
        if relevant.get(u, set()) - seen.get(u, set())
    ]


def main():
    parser = argparse.ArgumentParser(description="Part 2 disaggregated test evaluation")
    parser.add_argument("--data-dir", type=Path, default=Path("data/ml-1m"))
    parser.add_argument("--outputs-dir", type=Path, default=Path("outputs"))
    args = parser.parse_args()

    from surprise import Dataset, Reader
    from src.data_loader import load_movies, load_ratings
    from src import (
        content_based,
        item_based_kNN,
        matrix_factorization,
        most_popular,
        services,
        user_based_kNN,
    )
    from src.evaluation import evaluate_ranking
    from src.final_evaluation import SIMILARITY_METHODS, USER_SIMILARITY_METHODS

    # Read, do NOT refresh or edit, the validation-selected frozen settings.
    config_path = args.outputs_dir / "frozen_test_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing frozen configuration: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    required = {
        "knn", "user_based_knn", "mf", "content_based",
        "preferred_popularity", "popularity_variants",
        "cutoff", "relevance_threshold", "minimum_history",
    }
    if not required.issubset(config):
        raise ValueError(f"Incomplete frozen configuration: {sorted(required - set(config))}")
    k = int(config["cutoff"])
    threshold = int(config["relevance_threshold"])
    if k != 10:
        raise ValueError("This analysis is written for the project cutoff of 10.")

    # Exactly the split in run_all.py: stable sort, first 70%, next 10%, last 20%.
    ratings = load_ratings(args.data_dir)
    movies = load_movies(args.data_dir)
    order = np.argsort(ratings["Timestamp"].to_numpy(copy=True), kind="stable")
    ratings_sorted = ratings.iloc[order].reset_index(drop=True)
    train_end = int(len(ratings_sorted) * 0.70)
    validation_end = int(len(ratings_sorted) * 0.80)
    history = ratings_sorted.iloc[:validation_end].copy()  # train + validation
    test = ratings_sorted.iloc[validation_end:].copy()
    del ratings, ratings_sorted

    counts = history.groupby("UserID")["MovieID"].nunique()
    eligible = counts[counts >= int(config["minimum_history"])].index
    target_users = test.loc[test["UserID"].isin(eligible), "UserID"].unique()
    valid_users = evaluable_users(test, history, target_users, threshold)

    # User-activity boundary is computed ONLY from train + validation.
    # Compute median on the evaluable overall test users, matching the 1114-user check.
    activity = history.groupby("UserID")["MovieID"].size().reindex(valid_users)
    activity_cutoff = float(activity.median())
    low_users = activity.index[activity <= activity_cutoff].to_numpy()
    high_users = activity.index[activity > activity_cutoff].to_numpy()

    # Temporal boundary is the MEDIAN EVENT timestamp in the test set.
    # The two parts have equal numbers of ratings, NOT equal calendar duration.
    time_cutoff = float(test["Timestamp"].median())
    early_test = test.loc[test["Timestamp"] <= time_cutoff].copy()
    late_test = test.loc[test["Timestamp"] > time_cutoff].copy()

    print(f"Frozen settings loaded from {config_path}", flush=True)
    print(f"History ratings: {len(history):,}; test ratings: {len(test):,}", flush=True)
    print(f"Target users: {len(target_users)}; overall evaluable users: {len(valid_users)}", flush=True)
    print(
        f"Activity median: {activity_cutoff:g}; "
        f"low={len(low_users)}, high={len(high_users)}", flush=True
    )
    print(
        "Test time median (UTC): "
        f"{pd.to_datetime(time_cutoff, unit='s', utc=True)}; "
        f"early ratings={len(early_test):,}, late ratings={len(late_test):,}",
        flush=True,
    )

    # All five groups use the SAME frozen recommendations produced from history.
    # We change only the evaluated user population / held-out relevance labels.
    groups = [
        ("Overall", "All eligible users", target_users, test),
        ("User activity", "Low activity", low_users, test),
        ("User activity", "High activity", high_users, test),
        ("Test period", "Early test", target_users, early_test),
        ("Test period", "Late test", target_users, late_test),
    ]
    output_rows = []

    def add_model(model_label, predictions):
        predictions = top_ten(predictions, k)
        for dimension, group_name, users, heldout in groups:
            users = np.asarray(users)
            n_users = len(evaluable_users(heldout, history, users, threshold))
            metrics = evaluate_ranking(
                predictions, heldout, history, users, movies["MovieID"],
                cutoff=k, relevance_threshold=threshold,
            )
            output_rows.append({
                "Model": model_label,
                "Dimension": dimension,
                "Group": group_name,
                "Target users": len(users),
                "Evaluable users": n_users,
                "Held-out ratings": len(heldout),
                **metrics,
            })
            print(
                f"  {group_name}: users={n_users}, "
                f"NDCG@10={metrics['NDCG@10']:.6f}", flush=True
            )
        del predictions

    # 1. Selected Item-Based kNN baseline.
    print("\n[1/5] Item-Based kNN", flush=True)
    item_matrix = item_based_kNN.build_movie_user_matrix(history)
    item_settings = config["knn"]
    item_similarity = SIMILARITY_METHODS[item_settings["similarity"]](item_matrix)
    item_predictions = item_based_kNN.generate_prediction_scores(
        target_users, history, item_matrix, item_similarity,
        int(item_settings["k"]), catalogue_ids=movies["MovieID"], verbose=False,
    )
    add_model("Item-based kNN", item_predictions)
    del item_matrix, item_similarity, item_predictions

    # 2. Most Popular: use only the variant chosen on VALIDATION.
    print("\n[2/5] Most Popular (validation-selected variant)", flush=True)
    variant = next(
        (
            v for v in config["popularity_variants"]
            if v["name"] == config["preferred_popularity"]
        ),
        None,
    )
    if variant is None:
        raise ValueError("Frozen popularity variant is not in the saved list.")
    popularity = most_popular.compute_popularity(
        history, movies["MovieID"], variant["window_days"], variant["min_rating"]
    )
    popular_predictions = most_popular.generate_prediction_scores(
        target_users, history, popularity
    )
    add_model("Most popular", popular_predictions)
    del popular_predictions, popularity

    # 3. User-Based kNN.
    print("\n[3/5] User-Based kNN", flush=True)
    user_settings = config["user_based_knn"]
    user_matrix = user_based_kNN.build_user_movie_matrix(history)
    user_similarity = USER_SIMILARITY_METHODS[user_settings["similarity"]](user_matrix)
    user_predictions = user_based_kNN.generate_prediction_scores(
        target_users, history, user_matrix, user_similarity,
        int(user_settings["k"]), catalogue_ids=movies["MovieID"], verbose=False,
    )
    add_model("User-Based kNN", user_predictions)
    del user_matrix, user_similarity, user_predictions

    # 4. Matrix Factorization (Surprise SVD), fitted on train + validation.
    print("\n[4/5] Matrix Factorization", flush=True)
    surprise_data = Dataset.load_from_df(
        history[["UserID", "MovieID", "Rating"]], Reader(rating_scale=(1, 5))
    )
    surprise_trainset = surprise_data.build_full_trainset()
    movie_user_matrix = services.build_movie_user_matrix(history)
    movie_ids = movie_user_matrix.index.to_numpy()
    mask = movie_user_matrix.loc[:, target_users].isna().T.to_numpy()
    cold_movie_ids = pd.Index(movies["MovieID"]).unique().difference(
        movie_user_matrix.index, sort=False
    )
    mf_predictions = matrix_factorization.get_predictions(
        surprise_trainset, target_users, movie_ids, mask, cold_movie_ids,
        history["Rating"].mean(), **config["mf"],
    )
    add_model("Matrix factorization (SVD)", mf_predictions)
    del mf_predictions, movie_user_matrix, surprise_trainset, surprise_data, mask

    # 5. Self-implemented Content-Based model; batches match final_evaluation.py.
    print("\n[5/5] Content-Based", flush=True)
    content_settings = config["content_based"]
    content_features = content_based.build_item_features(
        movies, year_weight=content_settings["year_weight"]
    )
    content_profiles = content_based.build_user_profiles(
        history, content_features, weight_mode=content_settings["weight_mode"]
    )
    batches = []
    for start in range(0, len(target_users), 50):
        batch_users = target_users[start:start + 50]
        batch_predictions = content_based.generate_prediction_scores(
            batch_users, history, content_features, content_profiles
        )
        batches.append(top_ten(batch_predictions, k))
    content_predictions = pd.concat(batches, ignore_index=True)
    add_model("Content-Based", content_predictions)
    del content_predictions, content_features, content_profiles, batches

    results = pd.DataFrame(output_rows)

    # Safety check: for each model, OVERALL metrics must match the original final test.
    # This detects changed model code, candidate ranking, or accidental split changes.
    original_path = Path("results.csv")
    if not original_path.exists():
        raise FileNotFoundError(f"Expected original final test CSV: {original_path}")
    original = pd.read_csv(original_path)
    for model in results["Model"].unique():
        subset = original.loc[original["Model"] == model]
        if model == "Most popular":
            subset = subset.loc[subset["Variant / settings"] == variant["name"]]
        if len(subset) != 1:
            raise ValueError(f"Cannot find a unique original results.csv row for {model}.")
        overall = results.loc[
            (results["Model"] == model) & (results["Dimension"] == "Overall")
        ].iloc[0]
        for metric in ("Precision@10", "Recall@10", "NDCG@10", "Catalogue coverage"):
            recomputed = float(overall[metric])
            saved = float(subset.iloc[0][metric])
            if not np.isclose(recomputed, saved, atol=1e-9, rtol=0):
                raise AssertionError(
                    f"{model} overall {metric} differs from final result: "
                    f"now={recomputed}, saved={saved}. No output has been saved."
                )

    args.outputs_dir.mkdir(parents=True, exist_ok=True)
    output = args.outputs_dir / "disaggregated_results.csv"
    results.to_csv(output, index=False)
    print("\nAll five OVERALL results agree with results.csv.")
    print("\nNDCG@10 by model and group:")
    print(
        results.pivot(index="Model", columns="Group", values="NDCG@10")
        .round(6).to_string()
    )
    print(f"\nSaved {output}")


if __name__ == "__main__":
    main()
