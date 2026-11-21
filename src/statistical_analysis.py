import argparse
from pathlib import Path

import numpy as np
import pandas as pd


# Match the tie breaking used by src.evaluation.evaluate_ranking.
def top_k_predictions(predictions, cutoff):
    return (
        predictions.sort_values(
            ["UserID", "Prediction", "MovieID"],
            ascending=[True, False, True],
        )
        .groupby("UserID", sort=False)
        .head(cutoff)
        .reset_index(drop=True)
    )


# Calculate exactly the same per-user NDCG values that the shared evaluator averages.
def per_user_ndcg(top_predictions, held_out, history, target_users, cutoff, threshold):
    relevant = held_out.loc[held_out["Rating"] >= threshold]
    relevant_by_user = relevant.groupby("UserID")["MovieID"].agg(set).to_dict()
    seen_by_user = history.groupby("UserID")["MovieID"].agg(set).to_dict()
    recs_by_user = top_predictions.groupby("UserID")["MovieID"].agg(list).to_dict()
    discounts = 1.0 / np.log2(np.arange(2, cutoff + 2))

    rows = []
    for user in target_users:
        positives = relevant_by_user.get(user, set()) - seen_by_user.get(user, set())
        if not positives:
            # Match evaluate_ranking: exclude users without relevant unseen test items.
            continue

        recommendations = recs_by_user.get(user, [])
        hits = np.array([movie in positives for movie in recommendations], dtype=float)
        dcg = float((hits * discounts[:len(hits)]).sum())
        ideal_dcg = float(discounts[:min(cutoff, len(positives))].sum())
        rows.append({"UserID": int(user), "NDCG@10": dcg / ideal_dcg})

    return pd.DataFrame(rows, columns=["UserID", "NDCG@10"])


# A paired sign-flip permutation test for the mean difference, plus paired bootstrap CI.
def paired_inference(differences, seed=42, permutations=20000, bootstraps=10000):
    differences = np.asarray(differences, dtype=float)
    if not len(differences) or not np.isfinite(differences).all():
        raise ValueError("No valid paired user-level NDCG differences.")

    rng = np.random.default_rng(seed)
    observed = float(differences.mean())
    absolute_observed = abs(observed)
    extremes = 0

    # Monte Carlo two-sided paired sign-flip test.
    for start in range(0, permutations, 500):
        batch_size = min(500, permutations - start)
        signs = rng.choice(np.array([-1.0, 1.0]), size=(batch_size, len(differences)))
        permuted_means = (signs * differences).mean(axis=1)
        extremes += int(np.count_nonzero(np.abs(permuted_means) >= absolute_observed - 1e-15))
    p_value = (extremes + 1) / (permutations + 1)

    # Resample users as pairs, not model scores independently.
    boot_means = []
    for start in range(0, bootstraps, 500):
        batch_size = min(500, bootstraps - start)
        positions = rng.integers(0, len(differences), size=(batch_size, len(differences)))
        boot_means.extend(differences[positions].mean(axis=1))
    ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])
    return observed, float(ci_low), float(ci_high), float(p_value)


def main():
    parser = argparse.ArgumentParser(description="Part 2 paired user-level statistical comparison")
    parser.add_argument("--data-dir", type=Path, default=Path("data/ml-1m"))
    parser.add_argument("--results-dir", type=Path, default=Path("outputs"))
    args = parser.parse_args()

    # Import the same team modules used for the successful final test.
    from surprise import Dataset, Reader
    from src.data_loader import load_ratings, load_movies
    from src import most_popular, matrix_factorization, services
    from src.evaluation import evaluate_ranking
    from src.final_evaluation import freeze_configuration

    results_dir = args.results_dir
    # Do not refresh, rewrite, or otherwise select new hyperparameters.
    config = freeze_configuration(results_dir, refresh_config=False)
    cutoff = int(config["cutoff"])
    threshold = int(config["relevance_threshold"])
    if cutoff != 10:
        raise ValueError("This script currently requires the project's cutoff=10.")

    # Reproduce the exact chronological split in run_all.py.
    ratings = load_ratings(args.data_dir)
    movies = load_movies(args.data_dir)
    order = np.argsort(ratings["Timestamp"].to_numpy(copy=True), kind="stable")
    ratings_sorted = ratings.iloc[order].reset_index(drop=True)
    n = len(ratings_sorted)
    train_end = int(n * 0.70)
    validation_end = int(n * 0.80)
    history = pd.concat(
        [ratings_sorted.iloc[:train_end], ratings_sorted.iloc[train_end:validation_end]],
        ignore_index=True,
    )
    test = ratings_sorted.iloc[validation_end:].copy()
    counts = history.groupby("UserID")["MovieID"].nunique()
    eligible_users = counts[counts >= int(config["minimum_history"])].index
    target_users = test.loc[test["UserID"].isin(eligible_users), "UserID"].unique()
    print(f"Target users: {len(target_users)}; history ratings: {len(history):,}", flush=True)

    # Baseline selected by validation, NOT the highest test-set popularity variant.
    preferred_name = config["preferred_popularity"]
    preferred = next(
        (v for v in config["popularity_variants"] if v["name"] == preferred_name),
        None,
    )
    if preferred is None:
        raise ValueError("Frozen popularity variant was not found in config.")
    popularity = most_popular.compute_popularity(
        history, movies["MovieID"], preferred["window_days"], preferred["min_rating"]
    )
    print(f"Scoring frozen Most Popular: {preferred_name}", flush=True)
    pop_predictions = most_popular.generate_prediction_scores(target_users, history, popularity)
    pop_top10 = top_k_predictions(pop_predictions, cutoff)
    del pop_predictions
    pop_users = per_user_ndcg(pop_top10, test, history, target_users, cutoff, threshold)
    pop_macro = evaluate_ranking(
        pop_top10, test, history, target_users, movies["MovieID"],
        cutoff=cutoff, relevance_threshold=threshold,
    )["NDCG@10"]
    if not np.isclose(pop_users["NDCG@10"].mean(), pop_macro, atol=1e-12, rtol=0):
        raise AssertionError("Popularity per-user NDCG disagrees with team evaluator.")
    del pop_top10

    # Refit Surprise SVD using exactly the frozen MF settings on Train + Validation.
    print(f"Refitting frozen MF: {config['mf']}", flush=True)
    data = Dataset.load_from_df(
        history[["UserID", "MovieID", "Rating"]], Reader(rating_scale=(1, 5))
    )
    trainset = data.build_full_trainset()
    matrix = services.build_movie_user_matrix(history)
    movie_ids = matrix.index.to_numpy()
    mask = matrix.loc[:, target_users].isna().T.to_numpy()
    cold_movie_ids = pd.Index(movies["MovieID"]).unique().difference(matrix.index, sort=False)
    mf_predictions = matrix_factorization.get_predictions(
        trainset, target_users, movie_ids, mask, cold_movie_ids,
        history["Rating"].mean(), **config["mf"],
    )
    mf_top10 = top_k_predictions(mf_predictions, cutoff)
    del mf_predictions, matrix, data, trainset
    mf_users = per_user_ndcg(mf_top10, test, history, target_users, cutoff, threshold)
    mf_macro = evaluate_ranking(
        mf_top10, test, history, target_users, movies["MovieID"],
        cutoff=cutoff, relevance_threshold=threshold,
    )["NDCG@10"]
    if not np.isclose(mf_users["NDCG@10"].mean(), mf_macro, atol=1e-12, rtol=0):
        raise AssertionError("MF per-user NDCG disagrees with team evaluator.")
    del mf_top10

    # Audit both means against the original, untouched final-test results.csv.
    original = pd.read_csv("results.csv")
    pop_row = original.loc[
        (original["Model"] == "Most popular") &
        (original["Variant / settings"] == preferred_name)
    ]
    mf_row = original.loc[original["Model"] == "Matrix factorization (SVD)"]
    if len(pop_row) != 1 or len(mf_row) != 1:
        raise ValueError("The selected model rows were not found uniquely in results.csv.")
    pop_saved = float(pop_row.iloc[0]["NDCG@10"])
    mf_saved = float(mf_row.iloc[0]["NDCG@10"])
    if not (np.isclose(pop_saved, pop_macro, atol=1e-9, rtol=0)
            and np.isclose(mf_saved, mf_macro, atol=1e-9, rtol=0)):
        raise ValueError(
            "Recomputed per-user means differ from saved final test results. "
            "Do not use the statistics until this is investigated. "
            f"Popularity: {pop_macro} vs {pop_saved}; MF: {mf_macro} vs {mf_saved}."
        )

    # Pair observations by UserID, preserving only users eligible for both models.
    paired = (
        pop_users.rename(columns={"NDCG@10": "Popular_NDCG@10"})
        .merge(mf_users.rename(columns={"NDCG@10": "MF_NDCG@10"}),
               on="UserID", how="inner", validate="one_to_one")
        .sort_values("UserID")
        .reset_index(drop=True)
    )
    if len(paired) != len(pop_users) or len(paired) != len(mf_users):
        raise AssertionError("Different sets of evaluable users across models.")
    paired["Difference_MF_minus_Popular"] = (
        paired["MF_NDCG@10"] - paired["Popular_NDCG@10"]
    )
    mean_diff, ci_low, ci_high, p_value = paired_inference(
        paired["Difference_MF_minus_Popular"].to_numpy(),
        seed=42, permutations=20000, bootstraps=10000,
    )

    # Save only new analysis files; do not touch any model or frozen result.
    results_dir.mkdir(parents=True, exist_ok=True)
    paired.to_csv(results_dir / "statistical_per_user_ndcg.csv", index=False)
    summary = pd.DataFrame([{
        "Comparison": "MF (best new model on validation) minus Most Popular (best baseline on validation)",
        "User_count": len(paired),
        "MF_mean_NDCG@10": mf_macro,
        "Popular_mean_NDCG@10": pop_macro,
        "Mean_difference_MF_minus_Popular": mean_diff,
        "Bootstrap_95pct_CI_low": ci_low,
        "Bootstrap_95pct_CI_high": ci_high,
        "Two_sided_paired_sign_flip_p": p_value,
        "Permutation_count": 20000,
        "Bootstrap_count": 10000,
        "Random_seed": 42,
    }])
    summary.to_csv(results_dir / "statistical_paired_comparison.csv", index=False)
    print("\nPAIRED USER-LEVEL NDCG@10 COMPARISON")
    print(summary.to_string(index=False))
    print("\nMeans verified against the original results.csv.")
    print("Saved outputs/statistical_per_user_ndcg.csv")
    print("Saved outputs/statistical_paired_comparison.csv")


if __name__ == "__main__":
    main()
