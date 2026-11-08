import numpy as np


def evaluate_ranking(predictions, held_out, train, target_users, catalogue_ids, cutoff=10, relevance_threshold=4):
    relevant = held_out[held_out["Rating"] >= relevance_threshold]
    relevant_movies = relevant.groupby("UserID")["MovieID"].agg(set).to_dict()
    seen_movies = train.groupby("UserID")["MovieID"].agg(set).to_dict()

    top = (predictions.sort_values(["UserID", "Prediction", "MovieID"],
                                   ascending=[True, False, True])
           .groupby("UserID").head(cutoff))
    top_movies = top.groupby("UserID")["MovieID"].agg(list).to_dict()

    precision_scores = []
    recall_scores = []
    ndcg_scores = []
    recommended_movies = set()
    discounts = 1 / np.log2(np.arange(2, cutoff + 2))

    for user in target_users:
        positives = relevant_movies.get(user, set()) - seen_movies.get(user, set())
        if not positives:
            continue

        recommendations = top_movies.get(user, [])
        hits = np.array([movie in positives for movie in recommendations])

        precision_scores.append(hits.sum() / cutoff)
        recall_scores.append(hits.sum() / len(positives))
        dcg = (hits * discounts[:len(hits)]).sum()
        ideal_dcg = discounts[:min(cutoff, len(positives))].sum()
        ndcg_scores.append(dcg / ideal_dcg)
        recommended_movies.update(recommendations)

    return {
        f"Precision@{cutoff}": float(np.mean(precision_scores)) if precision_scores else np.nan,
        f"Recall@{cutoff}": float(np.mean(recall_scores)) if recall_scores else np.nan,
        f"NDCG@{cutoff}": float(np.mean(ndcg_scores)) if ndcg_scores else np.nan,
        "Catalogue coverage": (len(recommended_movies) / len(set(catalogue_ids))
                               if precision_scores else np.nan),
    }
