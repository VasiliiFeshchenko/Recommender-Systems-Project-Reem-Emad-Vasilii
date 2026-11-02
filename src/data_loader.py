# Load MovieLens ratings into a pandas DataFrame

import pandas as pd

def load_ratings(data_dir):
    ratings = pd.read_csv(
        data_dir / "ratings.dat",
        sep="::",
        engine="python",
        header=None,
        names=["UserID", "MovieID", "Rating", "Timestamp"]
    )
    return ratings

# Load MovieLens movie titles and genres into a DataFrame

def load_movies(data_dir):
    movies = pd.read_csv(
        data_dir / "movies.dat",
        sep="::",
        engine="python",
        header=None,
        names=["MovieID", "Title", "Genres"],
        encoding="latin-1" # this cause some movie titles contain special characters that the default UTF-8 decoding cannot read correctly
    )
    return movies

# Load MovieLens user information into a pandas DataFrame

def load_users(data_dir):
    users = pd.read_csv(
        data_dir / "users.dat",
        sep="::",
        engine="python",
        header=None,
        names=["UserID", "Gender", "Age", "Occupation", "ZipCode"]
    )
    return users