from pathlib import Path

import numpy as np
import pandas as pd
from surprise import Dataset, Reader, SVD

from src import services
from src.evaluation import evaluate_ranking


def get_predictions(
    trainset, users, movie_ids, unrated_mask, cold_movie_ids, cold_score,
    n_factors, n_epochs, lr_all, reg_all, random_state=1,
):
    """Fit SVD and score all unrated user-movie pairs in batches."""
    model = SVD(
        n_factors=n_factors,
        n_epochs=n_epochs,
        lr_all=lr_all,
        reg_all=reg_all,
        random_state=random_state,
    )
    model.fit(trainset)

    inner_users = np.array(
        [trainset.to_inner_uid(user_id) for user_id in users], dtype=np.intp,
    )
    inner_movies = np.array(
        [trainset.to_inner_iid(movie_id) for movie_id in movie_ids], dtype=np.intp,
    )

    # This is Surprise SVD's unclipped estimate for known users and movies:
    # global mean + user bias + movie bias + user/movie factor dot product.
    scores = (
        trainset.global_mean
        + model.bu[inner_users, None]
        + model.bi[inner_movies][None, :]
        + model.pu[inner_users] @ model.qi[inner_movies].T
    )

    user_positions, movie_positions = np.nonzero(unrated_mask)
    known_predictions = pd.DataFrame({
        "UserID": np.asarray(users)[user_positions],
        "MovieID": np.asarray(movie_ids)[movie_positions],
        "Prediction": scores[user_positions, movie_positions],
    })

    if len(cold_movie_ids) == 0 or len(users) == 0:
        return known_predictions

    cold_predictions = pd.DataFrame({
        "UserID": np.repeat(np.asarray(users), len(cold_movie_ids)),
        "MovieID": np.tile(np.asarray(cold_movie_ids), len(users)),
        "Prediction": cold_score,
    })
    return pd.concat([known_predictions, cold_predictions], ignore_index=True)

"""the function tests a total of 60 hyperparameter combinations and returns the best one"""
def test_hyperparameters(train, validation, movies):
    users = services.get_target_users(validation, train)
    movie_user_matrix = services.build_movie_user_matrix(train)
    movie_ids = movie_user_matrix.index.to_numpy()
    unrated_mask = movie_user_matrix.loc[:, users].isna().T.to_numpy()

    # Movies absent from training receive the same global-mean score as in kNN.
    cold_movie_ids = pd.Index(movies["MovieID"]).unique().difference(
        movie_user_matrix.index, sort=False
    )
    cold_score = train["Rating"].mean()

    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(train[["UserID", "MovieID", "Rating"]], reader)
    trainset = data.build_full_trainset()

    n_factors = [1, 5, 20, 50, 200]
    n_epochs = [ 20, 100]
    lr_all = [ 0.005, 0.05]
    reg_all = [0.01, 0.02, 0.1]

    validation_results = []
    for k in n_factors:
        for n in n_epochs:
            for lr in lr_all:
                for reg in reg_all:
                    hyperparameters_info = (
                        f"n_factors = {k}; n_epochs = {n}; "
                        f"learning rate = {lr}; regularization = {reg}."
                    )
                    predictions = get_predictions(
                        trainset, users, movie_ids, unrated_mask, cold_movie_ids,
                        cold_score, k, n, lr, reg,
                    )
                    metrics = evaluate_ranking(
                        predictions, validation, train, users, movies["MovieID"],
                        cutoff=10, relevance_threshold=4,
                    )
                    validation_results.append({
                        "Model": "matrix factorization using SVD",
                        "Paradigm": "Latent-factor collaborative filtering",
                        "Self-implemented?": "No",
                        "k": k,
                        "n": n,
                        "lr": lr,
                        "reg": reg,
                        **metrics,
                        "Seed": 1,
                        "Notes": "--",
                    })
                    metric_summary = " | ".join(
                        f"{name}: {value:.6f}" for name, value in metrics.items()
                    )
                    print(hyperparameters_info, "|", metric_summary, flush=True)

    results = pd.DataFrame(validation_results)
    results_dir = Path("outputs")
    results_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(results_dir / "MF_SVD_validation_results.csv", index=False)

    best_result = results.sort_values("NDCG@10", ascending=False).iloc[0]
    print("\nMatrix factorization validation comparison:")
    print(results[[
        "k", "n", "lr", "reg", "Precision@10", "Recall@10",
        "NDCG@10", "Catalogue coverage",
    ]].to_string(index=False))
    print(
        "\nBest configuration by validation NDCG@10: "
        f"n_factors={int(best_result['k'])}, n_epochs={int(best_result['n'])}, "
        f"learning rate={best_result['lr']}, regularization={best_result['reg']}"
    )
    print("Results saved to outputs/MF_SVD_validation_results.csv. Test set remains untouched.")
    return best_result
