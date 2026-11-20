import pandas as pd


def build_movie_user_matrix(train):
    return train.pivot(
        index="MovieID",
        columns="UserID",
        values="Rating",
    )

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

def get_unranked_movies_for_given_user(user_id, train):
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

def get_unrated_movies_for_all_users(train_movie_user_matrix):
    user_movie_matrix = train_movie_user_matrix.T
    unranked_movies = {}

    for user_id, ratings in user_movie_matrix.iterrows():
        unrated_movie_ids = ratings.index[ratings.isna()].to_numpy()
        unranked_movies[user_id] = unrated_movie_ids

    return unranked_movies
