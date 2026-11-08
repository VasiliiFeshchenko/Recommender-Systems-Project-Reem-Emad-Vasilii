import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity


#Dictionary for KNN hyperparameters
knn_param_grid = {
    "model__n_neighbors": [1, 2, 3, 5, 7, 9, 15, 25, 35, 50]
}

def build_movie_user_matrix(train):
    return train.pivot(
        index="MovieID",
        columns="UserID",
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


def generate_prediction_scores(target_users, train, train_movie_user_matrix, similarities, k, catalogue_ids=None, verbose=True):
    #get all the unique movies in training
    movie_ids = train_movie_user_matrix.index

    #covert the matrix to numPy. for faster calculations
    ratings = train_movie_user_matrix.to_numpy()

    #sort the values and then converting them to numpy. The order is especially important because numPY doesn't hold the lables. 
    similarity_values = similarities.loc[movie_ids, movie_ids].to_numpy()

    #after taking mean of each movie rating we sort (reindex) them to match the order in other data structures
    averages = train.groupby("MovieID")["Rating"].mean().reindex(movie_ids).to_numpy()

    # Preserve the candidate order used by get_unranked_movies.
    catalogue_positions = movie_ids.get_indexer(train["MovieID"].unique())

    #get all the movies that have no ratings
    cold_items = (pd.Index(catalogue_ids).unique().difference(movie_ids, sort=False)
                  if catalogue_ids is not None else pd.Index([]))

    
    global_mean = train["Rating"].mean()
    results = []
    batch_size = 256

    for user_count, user in enumerate(target_users, start=1):
        #get the column of a target user
        user_column = train_movie_user_matrix.columns.get_loc(user)
        #change the result to one dimentional array
        user_values = ratings[:, user_column]
        #find positions of the movies that target user has rated
        rated = np.flatnonzero(~np.isnan(user_values))
        #get the movies the user hasn't rated yet
        candidates = catalogue_positions[np.isnan(user_values[catalogue_positions])]
        #for the prediction calculation for each movie we need the difference of the average rating and user rating
        deviations = user_values[rated] - averages[rated]
        #for the prediction calculation for each movie we need the difference of the average rating and user rating
        scores = averages[candidates].copy()
        neighbor_count = min(k, len(rated))

        if neighbor_count:
            for start in range(0, len(candidates), batch_size):
                batch = candidates[start:start + batch_size]
                #get a smaller similarity matrix with only rated items
                weights = similarity_values[np.ix_(batch, rated)]
                #keep only positive weights
                weights = np.where(weights > 0, weights, 0.0)
                #Sort similarity list to get the closest neighbours; axis-1 specifies that we sort each row
                selected = np.argsort(-weights, axis=1, kind="stable")[:, :neighbor_count]
                #the actual similarity values at those selected positions
                selected_weights = np.take_along_axis(weights, selected, axis=1)
                numerator = (selected_weights * deviations[selected]).sum(axis=1)
                denominator = selected_weights.sum(axis=1)
                correction = np.divide(
                    numerator, denominator, out=np.zeros_like(numerator),
                    where=denominator > 0,
                )
                scores[start:start + len(batch)] += correction

        results.append(pd.DataFrame({
            "UserID": user,
            "MovieID": movie_ids.to_numpy()[candidates],
            "Prediction": scores,
        }))
        #if the moview is a cold item then we just insert global mean as it's rating.
        if len(cold_items):
            results.append(pd.DataFrame({
                "UserID": user, "MovieID": cold_items, "Prediction": global_mean,
            }))
        if verbose:
            print("User", user_count, "out of", len(target_users))

    if not results:
        return pd.DataFrame(columns=["UserID", "MovieID", "Prediction"])
    return pd.concat(results, ignore_index=True)
