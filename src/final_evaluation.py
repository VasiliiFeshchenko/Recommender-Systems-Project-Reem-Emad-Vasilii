"""Freeze validation choices, refit on train+validation, then evaluate on test."""

import json

import numpy as np
import pandas as pd

from src import item_based_kNN, most_popular, user_based_kNN, matrix_factorization, services

# Import the Content-Based model and its hyperparameter grid
from src import content_based
# Import the Content-Based model and its hyperparameter grid
from src import content_based
from src import mmr

from src.evaluation import evaluate_ranking
from surprise import Dataset, Reader


SIMILARITY_METHODS = {
    "Cosine": item_based_kNN.compute_cosine_similarity,
    "Pearson": item_based_kNN.compute_pearson_similarity,
    "Jaccard": item_based_kNN.compute_jaccard_similarity,
}


# Compute user-user similarities using the methods available in teammate's module.
USER_SIMILARITY_METHODS = {
    "Cosine": user_based_kNN.compute_cosine_similarity,
    "Pearson": user_based_kNN.compute_pearson_similarity,
    "Jaccard": user_based_kNN.compute_jaccard_similarity,
}


def select_user_knn_configuration(results_dir):
    # Read all 30 user-based kNN validation configurations.
    path = results_dir / "user_knn_validation_results.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing User-Based kNN validation results: {path}")
    results = pd.read_csv(path)

    # Verify the configurations match the 3 similarities x 10 k values grid.
    required = {"Similarity", "k", "NDCG@10"}
    if results.empty or not required.issubset(results.columns):
        raise ValueError(f"Invalid User-Based kNN validation CSV: {path}")
    expected = {
        (name, k)
        for name in USER_SIMILARITY_METHODS
        for k in user_based_kNN.knn_param_grid["model__n_neighbors"]
    }
    actual = set(zip(results["Similarity"], results["k"]))
    if (len(results) != len(expected) or actual != expected
            or not np.isfinite(results["NDCG@10"].to_numpy(dtype=float)).all()):
        raise ValueError(f"Incomplete User-Based kNN validation results: {path}")

    # Select using validation NDCG@10 only, resolving ties deterministically.
    best = results.sort_values(
        ["NDCG@10", "k", "Similarity"], ascending=[False, True, True]
    ).iloc[0]
    return {"similarity": str(best["Similarity"]), "k": int(best["k"])}


def select_mf_configuration(results_dir):
    # Read the 60 results saved by the matrix factorization teammate.
    path = results_dir / "MF_SVD_validation_results.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing MF validation results: {path}")
    results = pd.read_csv(path)

    # The teammate's CSV uses k/n/lr/reg for SVD hyperparameters.
    columns = ["k", "n", "lr", "reg", "Seed", "NDCG@10"]
    if (results.empty or len(results) != 60
            or not set(columns).issubset(results.columns)
            or not np.isfinite(results[columns].to_numpy(dtype=float)).all()
            or results.duplicated(subset=["k", "n", "lr", "reg", "Seed"]).any()):
        raise ValueError(f"MF validation results must contain 60 valid unique settings: {path}")

    # Choose the best setting from validation, never from test.
    best = results.sort_values("NDCG@10", ascending=False, kind="stable").iloc[0]
    return {
        "n_factors": int(best["k"]),
        "n_epochs": int(best["n"]),
        "lr_all": float(best["lr"]),
        "reg_all": float(best["reg"]),
        "random_state": int(best["Seed"]),
    }

def select_mmr_configuration(results_dir):
    """
    Select the best MMR configuration using the validation results.

    This function:
    1. Loads the MMR validation results CSV.
    2. Checks that all expected Lambda values are present.
    3. Validates that the results contain the required columns
       and valid NDCG@10 values.
    4. Ensures that the same base popularity configuration
       was used for every Lambda value.
    5. Selects the Lambda with the highest validation NDCG@10.

    If two configurations have the same NDCG@10, the larger
    Lambda value is preferred.

    Returns:
        dict: The selected Lambda value and its corresponding
              base Most Popular configuration.
    """
    path = results_dir / "mmr_validation_results.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"Missing MMR validation results: {path}"
        )

    results = pd.read_csv(path)

    # Expected Lambda values from 0.0 to 1.0
    expected_lambdas = {
        round(value, 1) for value in np.arange(0.0, 1.01, 0.1)
    }

    required = {"Lambda", "NDCG@10", "Base popularity variant"}

    if results.empty or not required.issubset(results.columns):
        raise ValueError(f"Invalid MMR validation CSV: {path}")

    actual_lambdas = set(
        results["Lambda"].astype(float).round(1)
    )

    if (
        len(results) != len(expected_lambdas)
        or actual_lambdas != expected_lambdas
        or not np.isfinite(results["NDCG@10"].to_numpy(dtype=float)).all()
        or results["Base popularity variant"].nunique() != 1
    ):
        raise ValueError(
            f"Incomplete or invalid MMR validation results: {path}"
        )

    # Select the best Lambda using validation NDCG@10.
    # If tied, prefer the larger Lambda (more relevance).
    best = results.sort_values(
        ["NDCG@10", "Lambda"],
        ascending=[False, False],
        kind="stable"
    ).iloc[0]

    return {
        "lambda": float(best["Lambda"]),
        "base_popularity_variant": str(
            best["Base popularity variant"]
        ),
    }

def freeze_configuration(results_dir, refresh_config=False):
    path = results_dir / "frozen_test_config.json"
    if path.exists() and not refresh_config:
        # Older frozen files omit the new Part 2 models; rebuild them from validation.
        saved = json.loads(path.read_text(encoding="utf-8"))
        required_keys = {
            "knn", "preferred_popularity", "popularity_variants",
            "content_based", "user_based_knn", "mf","mmr",
            "cutoff", "relevance_threshold", "minimum_history",
        }
        if required_keys.issubset(saved):
            print(f"Reusing frozen choices from {path}", flush=True)
            return saved
        print("Frozen file lacks new models. Rebuilding using saved validation CSVs.", flush=True)

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
    # Locate the saved Content-Based validation results
    content_path = results_dir / "content_based_validation_results.csv"

    # Ensure the validation results exist
    if not content_path.exists():
        raise FileNotFoundError("Content-Based validation results are missing.")

    # Load the eight evaluated configurations
    content_results = pd.read_csv(content_path)

    # Define the eight configurations we expected to evaluate
    expected_content = {
        (mode, float(year))
        for mode in content_based.WEIGHT_MODES
        for year in content_based.YEAR_WEIGHTS
    }

    # Collect the configurations actually present in the CSV
    actual_content = set(
        zip(content_results["Weight mode"], content_results["Year weight"])
    )

    # Ensure all eight configurations were evaluated exactly once
    if (
        len(content_results) != len(expected_content)
        or actual_content != expected_content
        or not np.isfinite(content_results["NDCG@10"].to_numpy()).all()
    ):
        raise ValueError("Content-Based validation results are incomplete or invalid.")

    # Select the best configuration using validation NDCG
    best_content = content_results.sort_values(
        ["NDCG@10", "Weight mode", "Year weight"],
        ascending=[False, True, True]
    ).iloc[0]

    config = {
        "selection_metric": "validation NDCG@10",
        "knn": {"similarity": best_knn["Similarity"], "k": int(best_knn["k"])},
         # Save the selected Content-Based configuration
        "content_based": {
            "weight_mode": str(best_content["Weight mode"]),
            "year_weight": float(best_content["Year weight"]),
        },

        # Freeze the two additional Part 2 models from their validation CSVs.
        "user_based_knn": select_user_knn_configuration(results_dir),
        "mmr": select_mmr_configuration(results_dir),
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
    config = freeze_configuration(results_dir, refresh_config=refresh_config)
    history = pd.concat([train, validation], ignore_index=True)
    history_counts = history.groupby("UserID")["MovieID"].nunique()
    eligible = history_counts[history_counts >= config["minimum_history"]].index
    target_users = test.loc[test["UserID"].isin(eligible), "UserID"].unique()
    print(f"Final evaluation: refit on {len(history):,} ratings; {len(target_users)} target users", flush=True)
    print(f"Frozen item-kNN: {config['knn']}; user-kNN: {config['user_based_knn']}", flush=True)
    print(f"Frozen MF: {config['mf']}; Content-Based: {config['content_based']}", flush=True)
    print(f"Frozen MMR: {config['mmr']}", flush=True)

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
    matrix = item_based_kNN.build_movie_user_matrix(history)
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
                 + "; cold items use history global mean; no neighbors use item mean",
    })
    print("kNN test metrics:", metrics, flush=True)
    del predictions, similarities, matrix

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
                     + "; recent windows include timestamps>=end-days*86400; zero count for unobserved items",
        })
        del predictions

    # ------------------------------------------------
    # User-Based kNN: refit on Train + Validation and score Test users.
    # ------------------------------------------------
    user_settings = config["user_based_knn"]
    user_matrix = user_based_kNN.build_user_movie_matrix(history)
    user_similarities = USER_SIMILARITY_METHODS[user_settings["similarity"]](user_matrix)
    user_predictions = user_based_kNN.generate_prediction_scores(
        target_users, history, user_matrix, user_similarities,
        user_settings["k"], catalogue_ids=movies["MovieID"], verbose=False,
    )
    user_metrics = evaluate_ranking(
        user_predictions, test, history, target_users, movies["MovieID"],
        cutoff=config["cutoff"], relevance_threshold=config["relevance_threshold"],
    )
    rows.append({
        "Model": "User-Based kNN",
        "Variant / settings": "User-mean-centered; positive neighbors; full catalogue",
        "Similarity": user_settings["similarity"],
        "k": user_settings["k"], **user_metrics, "Seed": 0,
        "Notes": common_notes + "; deterministic model, seed unused; "
                 "cold items use global mean; fallback movie means",
    })
    print("User-Based kNN test metrics:", user_metrics, flush=True)
    del user_predictions, user_similarities, user_matrix

    # ------------------------------------------------
    # Matrix Factorization (Surprise SVD): refit on Train + Validation.
    # ------------------------------------------------
    mf_settings = config["mf"]
    print(f"Fitting Matrix Factorization using frozen settings: {mf_settings}", flush=True)
    mf_data = Dataset.load_from_df(
        history[["UserID", "MovieID", "Rating"]], Reader(rating_scale=(1, 5))
    )
    mf_trainset = mf_data.build_full_trainset()

    # Use the same full-catalogue candidate policy as the teammate's validation.
    movie_user_matrix = services.build_movie_user_matrix(history)
    movie_ids = movie_user_matrix.index.to_numpy()
    unrated_mask = movie_user_matrix.loc[:, target_users].isna().T.to_numpy()
    cold_movie_ids = pd.Index(movies["MovieID"]).unique().difference(
        movie_user_matrix.index, sort=False
    )
    mf_predictions = matrix_factorization.get_predictions(
        mf_trainset, target_users, movie_ids, unrated_mask,
        cold_movie_ids, history["Rating"].mean(), **mf_settings,
    )
    mf_metrics = evaluate_ranking(
        mf_predictions, test, history, target_users, movies["MovieID"],
        cutoff=config["cutoff"], relevance_threshold=config["relevance_threshold"],
    )
    rows.append({
        "Model": "Matrix factorization (SVD)",
        "Variant / settings": ", ".join(
            f"{key}={mf_settings[key]}"
            for key in ("n_factors", "n_epochs", "lr_all", "reg_all")
        ),
        "Similarity": "", "k": "", **mf_metrics,
        "Seed": mf_settings["random_state"],
        "Notes": common_notes + "; SVD, cold items use history global mean; "
                 "unclipped factorization scores",
    })
    print("Matrix Factorization test metrics:", mf_metrics, flush=True)
    del mf_predictions, mf_trainset, mf_data, movie_user_matrix

    # Get the Content-Based settings selected during validation
    content_settings = config["content_based"]

    # Build movie features using the selected year weight
    content_features = content_based.build_item_features(
        movies, year_weight=content_settings["year_weight"]
    )

    # Rebuild user profiles using Train + Validation
    content_profiles = content_based.build_user_profiles(
        history,
        content_features,
        weight_mode=content_settings["weight_mode"]
    )

    # Store the top-ten recommendations from each batch
    content_batches = []

    # Process target users in batches of 50
    for start in range(0, len(target_users), 50):

        # Select the current group of users
        batch_users = target_users[start:start + 50]

        # Calculate scores for their unseen movies
        batch_predictions = content_based.generate_prediction_scores(
            batch_users, history, content_features, content_profiles
        )

        # Rank movies by similarity, breaking ties by MovieID
        batch_predictions = batch_predictions.sort_values(
            ["UserID", "Prediction", "MovieID"],
            ascending=[True, False, True]
        )

        # Keep the best ten movies for each user
        content_batches.append(
            batch_predictions.groupby("UserID").head(config["cutoff"])
        )

    # Combine predictions for all target users
    content_predictions = pd.concat(content_batches, ignore_index=True)

    # Evaluate Content-Based using the same team evaluation function
    content_metrics = evaluate_ranking(
        content_predictions,
        test,
        history,
        target_users,
        movies["MovieID"],
        cutoff=config["cutoff"],
        relevance_threshold=config["relevance_threshold"]
    )

    # Add Content-Based results to the shared final results table
    rows.append({
        "Model": "Content-Based",
        "Variant / settings": (
            f"weight_mode={content_settings['weight_mode']}; "
            f"year_weight={content_settings['year_weight']}"
        ),
        "Similarity": "Cosine",
        "k": "",
        **content_metrics,
        "Seed": 0,
        "Notes": common_notes + "; self-implemented genre/year profile matching",
    })

    # Display Content-Based final metrics
    print("Content-Based final test metrics:", content_metrics, flush=True)

    # ------------------------------------------------
    # MMR: Final Test Evaluation
    # ------------------------------------------------

    mmr_settings = config["mmr"]

    # Find the Most Popular variant selected during validation
    base_variant = next(
        variant for variant in config["popularity_variants"]
        if variant["name"] == mmr_settings["base_popularity_variant"]
    )

    print(
        f"Evaluating MMR with frozen settings: {mmr_settings}",
        flush=True
    )

    # Generate Most Popular candidates and re-rank them using MMR
    mmr_predictions = mmr.generate_mmr_recommendations(
        target_users=target_users,
        train=history,
        catalogue_ids=movies["MovieID"],
        movies=movies,
        k=config["cutoff"],
        lambda_=mmr_settings["lambda"],
        window_days=base_variant["window_days"],
        min_rating=base_variant["min_rating"],
    )

    # Evaluate MMR using the existing team evaluation function
    mmr_metrics = evaluate_ranking(
        mmr_predictions,
        test,
        history,
        target_users,
        movies["MovieID"],
        cutoff=config["cutoff"],
        relevance_threshold=config["relevance_threshold"],
    )

    # Add MMR to the final results table
    rows.append({
        "Model": "MMR",
        "Variant / settings": (
            f"Base=Most Popular ({base_variant['name']}); "
            f"lambda={mmr_settings['lambda']}"
        ),
        "Similarity": "Cosine (genres)",
        "k": config["cutoff"],
        **mmr_metrics,
        "Seed": 0,
        "Notes": (
            common_notes
            + "; Most Popular candidate generation; "
            + "binary 18-genre vectors; cosine similarity; "
            + "greedy MMR re-ranking; "
            + f"lambda={mmr_settings['lambda']}; "
            + "lambda selected using validation NDCG@10"
        ),
    })

    print("MMR final test metrics:", mmr_metrics, flush=True)

    del mmr_predictions


    results = pd.DataFrame(rows)
    # Write only after all final evaluations finish; preserve validation CSVs.
    results.to_csv("results.csv", index=False)
    print("\nFinal test results:")
    print(results[["Model", "Variant / settings", "Similarity", "k", "Precision@10",
                   "Recall@10", "NDCG@10", "Catalogue coverage"]].to_string(index=False))
    print("Saved results.csv. Model choices remain the ones selected on validation.")
