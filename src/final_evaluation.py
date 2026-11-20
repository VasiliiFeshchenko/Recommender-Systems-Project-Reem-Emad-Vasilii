"""Freeze validation choices, refit on train+validation, then evaluate on test."""

import json

import numpy as np
import pandas as pd

from src import item_based_kNN, most_popular, services
from src.evaluation import evaluate_ranking


SIMILARITY_METHODS = {
    "Cosine": item_based_kNN.compute_cosine_similarity,
    "Pearson": item_based_kNN.compute_pearson_similarity,
    "Jaccard": item_based_kNN.compute_jaccard_similarity,
}


def select_mf_configuration(results_dir):
    path = results_dir / "MF_SVD_validation_results.csv"
    if not path.exists():
        raise ValueError("MF validation results are missing. Run python run_all.py --validation-only first.")

    results = pd.read_csv(path)
    columns = ["k", "n", "lr", "reg", "Seed", "NDCG@10"]
    if (results.empty or not set(columns).issubset(results.columns)
            or not np.isfinite(results[columns].to_numpy(dtype=float)).all()):
        raise ValueError(f"{path} must contain valid numeric SVD settings and validation NDCG@10.")

    best = results.sort_values("NDCG@10", ascending=False, kind="stable").iloc[0]
    return {
        "n_factors": int(best["k"]),
        "n_epochs": int(best["n"]),
        "lr_all": float(best["lr"]),
        "reg_all": float(best["reg"]),
        "random_state": int(best["Seed"]),
    }


def freeze_configuration(results_dir, refresh_config=False):
    path = results_dir / "frozen_test_config.json"
    if path.exists() and not refresh_config:
        config = json.loads(path.read_text(encoding="utf-8"))
        if "mf" not in config:
            config["mf"] = select_mf_configuration(results_dir)
            path.write_text(json.dumps(config, indent=2), encoding="utf-8")
            print(f"Added the MF validation choice to {path}", flush=True)
        print(f"Reusing frozen choices from {path}", flush=True)
        return config

    knn_path = results_dir / "knn_validation_results.csv"
    popularity_path = results_dir / "popularity_validation_results.csv"
    if not knn_path.exists() or not popularity_path.exists():
        raise ValueError("Complete the validation run first: python run_all.py --data-dir data/ml-1m --validation-only")
    knn_results = pd.read_csv(knn_path)
    popularity_results = pd.read_csv(popularity_path)

    expected_knn = {(name, k) for name in SIMILARITY_METHODS
                    for k in item_based_kNN.knn_param_grid["model__n_neighbors"]}
    actual_knn = set(zip(knn_results["Similarity"], knn_results["k"]))
    variants = [
        {"name": f"{name}; count ratings>={threshold}",
         "window_days": days, "min_rating": threshold}
        for name, days in most_popular.POPULARITY_WINDOWS
        for threshold in most_popular.RATING_THRESHOLDS
    ]
    expected_popularity = {variant["name"] for variant in variants}
    if (actual_knn != expected_knn or len(knn_results) != len(expected_knn)
            or set(popularity_results["Variant / settings"]) != expected_popularity
            or len(popularity_results) != len(expected_popularity)):
        raise ValueError("Validation CSVs are incomplete or outdated. Finish all 30 kNN and 9 popularity runs first.")
    if not (np.isfinite(knn_results["NDCG@10"]).all()
            and np.isfinite(popularity_results["NDCG@10"]).all()):
        raise ValueError("Validation NDCG must be available before selecting final models.")

    best_knn = knn_results.sort_values(
        ["NDCG@10", "k", "Similarity"], ascending=[False, True, True]
    ).iloc[0]
    # Reindex to the predefined variant order, which also resolves equal NDCG scores.
    popularity_results = popularity_results.set_index("Variant / settings").loc[
        [variant["name"] for variant in variants]
    ]
    best_popularity = popularity_results["NDCG@10"].idxmax()
    config = {
        "selection_metric": "validation NDCG@10",
        "knn": {"similarity": best_knn["Similarity"], "k": int(best_knn["k"])},
        "mf": select_mf_configuration(results_dir),
        "preferred_popularity": best_popularity,
        "popularity_variants": variants,
        "cutoff": 10,
        "relevance_threshold": 4,
        "minimum_history": 5,
    }
    # Save before any model is refitted or evaluated on test.
    results_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"Choices frozen in {path}", flush=True)
    return config


def run_final_evaluation(train, validation, test, movies, results_dir, refresh_config=False):
    from surprise import Dataset, Reader

    from src import matrix_factorization

    config = freeze_configuration(results_dir, refresh_config=refresh_config)
    history = pd.concat([train, validation], ignore_index=True)
    history_counts = history.groupby("UserID")["MovieID"].nunique()
    eligible = history_counts[history_counts >= config["minimum_history"]].index
    target_users = test.loc[test["UserID"].isin(eligible), "UserID"].unique()
    print(f"Final evaluation: refit on {len(history):,} ratings; {len(target_users)} target users", flush=True)
    print(f"Frozen kNN: {config['knn']}; preferred popularity: {config['preferred_popularity']}", flush=True)

    common_notes = (
        "FINAL TEST; choices frozen from validation; fitted on train+validation; "
        f"evaluation relevance rating>={config['relevance_threshold']}; "
        f"history>={config['minimum_history']}; full catalogue; exclude all history-rated movies; "
        "macro accuracy; users without relevant unseen test movies excluded from accuracy "
        "and coverage; ties by MovieID ascending"
    )
    rows = []
    similarity_name = config["knn"]["similarity"]
    k = config["knn"]["k"]
    matrix = services.build_movie_user_matrix(history)
    print(f"Computing {similarity_name} similarities from train+validation...", flush=True)
    similarities = SIMILARITY_METHODS[similarity_name](matrix)
    predictions = item_based_kNN.generate_prediction_scores(
        target_users, history, matrix, similarities, k,
        catalogue_ids=movies["MovieID"], verbose=False,
    )
    metrics = evaluate_ranking(
        predictions, test, history, target_users, movies["MovieID"],
        cutoff=config["cutoff"], relevance_threshold=config["relevance_threshold"],
    )
    similarity_note = {
        "Cosine": "zero-filled history vectors",
        "Pearson": "shared raters; minimum 2; undefined correlations set to zero",
        "Jaccard": "overlap of history rater sets; all observed ratings count",
    }[similarity_name]
    rows.append({
        "Model": "Item-based kNN",
        "Variant / settings": "Item-mean-centered; positive neighbors; full catalogue",
        "Similarity": similarity_name, "k": k, **metrics, "Seed": 0,
        "Notes": common_notes + "; " + similarity_note
                 + "; cold items use history global mean; no neighbors use item mean; seed unused",
    })
    print("kNN test metrics:", metrics, flush=True)
    del predictions, similarities

    mf_settings = config["mf"]
    print(f"Refitting Surprise SVD on train+validation: {mf_settings}", flush=True)
    data = Dataset.load_from_df(history[["UserID", "MovieID", "Rating"]], Reader(rating_scale=(1, 5)))
    trainset = data.build_full_trainset()
    cold_movie_ids = pd.Index(movies["MovieID"]).unique().difference(matrix.index, sort=False)
    predictions = matrix_factorization.get_predictions(
        trainset, target_users, matrix.index.to_numpy(),
        matrix.loc[:, target_users].isna().T.to_numpy(),
        cold_movie_ids, history["Rating"].mean(), **mf_settings,
    )
    metrics = evaluate_ranking(
        predictions, test, history, target_users, movies["MovieID"],
        cutoff=config["cutoff"], relevance_threshold=config["relevance_threshold"],
    )
    rows.append({
        "Model": "Matrix factorization (SVD)",
        "Variant / settings": ", ".join(
            f"{name}={mf_settings[name]}" for name in ("n_factors", "n_epochs", "lr_all", "reg_all")
        ),
        "Similarity": "", "k": "", **metrics, "Seed": mf_settings["random_state"],
        "Notes": common_notes + "; cold items use history global mean; unclipped SVD scores",
    })
    print("MF test metrics:", metrics, flush=True)
    del data, trainset, predictions, matrix

    # Report all nine predeclared variants; never choose a winner from test scores.
    for variant in config["popularity_variants"]:
        print(f"Evaluating on test: {variant['name']}", flush=True)
        popularity = most_popular.compute_popularity(
            history, movies["MovieID"], variant["window_days"], variant["min_rating"],
        )
        predictions = most_popular.generate_prediction_scores(target_users, history, popularity)
        metrics = evaluate_ranking(
            predictions, test, history, target_users, movies["MovieID"],
            cutoff=config["cutoff"], relevance_threshold=config["relevance_threshold"],
        )
        rows.append({
            "Model": "Most popular", "Variant / settings": variant["name"],
            "Similarity": "", "k": "", **metrics, "Seed": 0,
            "Notes": common_notes
                     + f"; preferred by validation={variant['name'] == config['preferred_popularity']}"
                     + f"; window ends at train+validation max timestamp={int(history['Timestamp'].max())}"
                     + "; recent windows include timestamps>=end-days*86400; zero count for unobserved items; seed unused",
        })
        del predictions

    results = pd.DataFrame(rows)
    # Write only after all final evaluations finish; preserve validation CSVs.
    results.to_csv("results.csv", index=False)
    print("\nFinal test results:")
    print(results[["Model", "Variant / settings", "Similarity", "k", "Precision@10",
                   "Recall@10", "NDCG@10", "Catalogue coverage"]].to_string(index=False))
    print("Saved results.csv. Model choices remain the ones selected on validation.")
