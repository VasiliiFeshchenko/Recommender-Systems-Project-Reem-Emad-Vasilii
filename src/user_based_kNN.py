import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity


#Dictionary for KNN hyperparameters
knn_param_grid = {
    "model__n_neighbors": [1, 2, 3, 5, 7, 9, 15, 25, 35, 50]
}

def build_user_movie_matrix(train):
    return train.pivot(
        index="UserID",
        columns="MovieID",
        values="Rating",
    )

def compute_cosine_similarity(train_movie_user_matrix):

    #this fills missing values in the matrix with zeroes. This is a design choise -- we compare ull vectors
    ratings_for_cosine = train_movie_user_matrix.fillna(0)

    similarities = cosine_similarity(ratings_for_cosine)

    return pd.DataFrame(
        similarities,
        index=train_movie_user_matrix.index,
        columns=train_movie_user_matrix.index,
    )

def compute_pearson_similarity(train_movie_user_matrix):
    # corr compares columns, so transpose to make each movie a column.
    # Missing ratings stay NaN: each pair uses only users who rated both movies.
    similarities = train_movie_user_matrix.T.corr(method="pearson", min_periods=2)
    # Fewer than two shared raters, or constant shared ratings: no usable similarity.
    return similarities.fillna(0.0)


def compute_jaccard_similarity(train_movie_user_matrix):
    # A 1 means the user rated the movie, regardless of the rating value.
    # Use numeric sparse entries so multiplication counts shared users.
    rated = csr_matrix(train_movie_user_matrix.notna().to_numpy(), dtype=np.float64)
    intersection = (rated @ rated.T).toarray()
    counts = np.asarray(rated.sum(axis=1)).ravel()
    union = counts[:, None] + counts[None, :] - intersection
    similarities = np.divide(
        intersection, union, out=np.zeros_like(intersection), where=union > 0,
    )
    return pd.DataFrame(
        similarities,
        index=train_movie_user_matrix.index,
        columns=train_movie_user_matrix.index,
    )


def find_neighbors(movie_id, user_ratings, similarities, k):
    candidate_similarities = similarities.loc[
        movie_id, user_ratings.index
    ]

    neighbors = pd.DataFrame({
        "rating": user_ratings,
        "similarity": candidate_similarities,
    })

    neighbors = neighbors[neighbors["similarity"] > 0]

    return neighbors.nlargest(k, "similarity")

#same method, but it uses NumPy to speed up calculations
def find_neighbors_using_np(movie_id, user_ratings, similarities, k):
    similarity_values = similarities.loc[movie_id, user_ratings.index].to_numpy()

    positive_positions = np.flatnonzero(similarity_values > 0)

    order = np.argsort(
        -similarity_values[positive_positions],
        kind="stable",
    )

    selected = positive_positions[order[:k]]

    return pd.DataFrame({
        "rating": user_ratings.to_numpy()[selected],
        "similarity": similarity_values[selected],
    }, index=user_ratings.index[selected])    
    
def get_target_users(validation, train):

    #Groups training rows by user, then count the distinct movies each user rated; output is a (n,) series with user ids as rows and rating counts as columns.
    training_counts = train.groupby("UserID")["MovieID"].nunique()

    #filter users who rated 5 or more movies; users with too little history cannot be accurately predicted. This is a design choice.
    eligible_users = training_counts[training_counts >= 5].index

    #produce an array of users from validation who are eligible (they rated 5 or more movies in training)
    # validation["UserID"].isin(eligible_users) produces True for validation rows belonging to eligible users
    # .loc[..., "UserID"] selects those rows and only their UserID column
    # .unique() removes repeated IDs, since one user may have multiple validation ratings
    validation_users = validation.loc[
        validation["UserID"].isin(eligible_users), "UserID"
    ].unique()

    return validation_users

def get_unranked_movies(user_id, train):
    all_movies = train["MovieID"].unique()

    # get the movies this user already rated in training
    # train["UserID"] == user_id identifies rows belonging to the target user
    #  .loc[..., "MovieID"] selects the movie IDs from those rows
    # .unique() removes duplicates.
    rated_movies = train.loc[
        train["UserID"] == user_id, "MovieID"
    ].unique()

    #subtract all the rated movies
    return pd.Index(all_movies).difference(rated_movies, sort=False).to_numpy()

def get_prediction(movie_avg, neighbours, user_deviations):
    denominator = neighbours["similarity"].abs().sum()

    if denominator == 0:
        return movie_avg

    deviations = user_deviations.loc[neighbours.index]
    numerator = (neighbours["similarity"] * deviations).sum()

    return movie_avg + numerator / denominator


def generate_prediction_scores(
    target_users,
    train,
    user_movie_matrix,
    similarities,
    k,
    catalogue_ids=None,
    verbose=False,
):
    """Rank unseen movies using the target user's nearest training users."""
    user_ids = user_movie_matrix.index
    movie_ids = user_movie_matrix.columns
    ratings = user_movie_matrix.to_numpy(dtype=float)

    user_means = (
        train.groupby("UserID")["Rating"]
        .mean()
        .reindex(user_ids)
        .to_numpy()
    )
    movie_means = train.groupby("MovieID")["Rating"].mean()
    global_mean = train["Rating"].mean()

    if catalogue_ids is None:
        catalogue = pd.Index(movie_ids).unique()
    else:
        catalogue = pd.Index(catalogue_ids).unique()

    results = []

    for user_number, user_id in enumerate(target_users, start=1):
        if user_id not in user_ids:
            continue

        target_position = user_ids.get_loc(user_id)
        target_mean = user_means[target_position]

        # Keep the k most similar positive-similarity users, excluding self.
        neighbor_similarities = similarities.loc[user_id].drop(index=user_id)
        neighbors = neighbor_similarities[neighbor_similarities > 0].nlargest(k)

        # Use movie means when neighbors have no rating for a movie,
        # and the training global mean for catalogue items unseen in training.
        scores = movie_means.reindex(catalogue).fillna(global_mean).astype(float)

        if not neighbors.empty:
            neighbor_positions = user_ids.get_indexer(neighbors.index)
            neighbor_ratings = ratings[neighbor_positions, :]
            observed = ~np.isnan(neighbor_ratings)
            neighbor_means = user_means[neighbor_positions]

            weights = neighbors.to_numpy()[:, None] * observed
            deviations = np.where(
                observed,
                neighbor_ratings - neighbor_means[:, None],
                0.0,
            )

            denominator = weights.sum(axis=0)
            numerator = (weights * deviations).sum(axis=0)
            adjustments = np.divide(
                numerator,
                denominator,
                out=np.zeros_like(numerator),
                where=denominator > 0,
            )

            movies_with_neighbor_ratings = movie_ids[denominator > 0]
            predicted_values = target_mean + adjustments[denominator > 0]
            prediction_by_movie = pd.Series(
                predicted_values,
                index=movies_with_neighbor_ratings,
            )
            available_movies = movies_with_neighbor_ratings.intersection(catalogue)
            scores.loc[available_movies] = prediction_by_movie.reindex(available_movies)

        # Do not recommend movies already rated by this user in training.
        seen_movies = user_movie_matrix.loc[user_id].dropna().index
        scores = scores.drop(index=seen_movies, errors="ignore")
        top_movies = scores.nlargest(10)

        results.append(pd.DataFrame({
            "UserID": user_id,
            "MovieID": top_movies.index,
            "Prediction": top_movies.to_numpy(),
        }))

        if verbose:
            print(f"User {user_number} of {len(target_users)}")

    if not results:
        return pd.DataFrame(columns=["UserID", "MovieID", "Prediction"])

    return pd.concat(results, ignore_index=True)
