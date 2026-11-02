import pandas as pd
import matplotlib.pyplot as plt
import math
# Calculate basic statistics about the MovieLens dataset

def dataset_overview(ratings, movies, users):

    n_users = ratings["UserID"].nunique()
    n_items = ratings["MovieID"].nunique()
    n_interactions = len(ratings)

    density = (n_interactions / (n_users * n_items)) * 100

    return {
        "users": n_users,
        "total_movies": len(movies),
        "rated_movies": n_items,
        "interactions": n_interactions,
        "density": density
    }

# Calculate the number of ratings per user and per movie

def interaction_counts(ratings):
    ratings_per_user = ratings.groupby("UserID").size()
    ratings_per_movie = ratings.groupby("MovieID").size()

    return ratings_per_user, ratings_per_movie

# Calculate the distribution of rating values from 1 to 5

def rating_distribution(ratings):
    rating_counts = ratings["Rating"].value_counts().sort_index()

    return rating_counts

# Calculate monthly rating activity from Unix timestamps

def monthly_rating_activity(ratings):

    dates = pd.to_datetime(ratings["Timestamp"], unit="s", utc=True)

    monthly_activity = (
        ratings.assign(Date=dates)
        .set_index("Date")
        .resample("MS")
        .size()
    )

    return monthly_activity

# Generate and save the movie long-tail distribution plot

def plot_movie_long_tail(ratings_per_movie, figures_dir):

    figures_dir.mkdir(parents=True, exist_ok=True)

    sorted_counts = ratings_per_movie.sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(range(1, len(sorted_counts) + 1), sorted_counts.values)

    ax.set_title("Long-Tail Distribution of Movie Ratings")
    ax.set_xlabel("Movies Ranked by Number of Ratings")
    ax.set_ylabel("Number of Ratings")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(figures_dir / "movie_long_tail.png", dpi=300)

    plt.close(fig)

# Generate and save the user long-tail distribution plot

def plot_user_long_tail(ratings_per_user, figures_dir):

    figures_dir.mkdir(parents=True, exist_ok=True)

    sorted_counts = ratings_per_user.sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(range(1, len(sorted_counts) + 1), sorted_counts.values)

    ax.set_title("Long-Tail Distribution of User Ratings")
    ax.set_xlabel("Users Ranked by Number of Ratings")
    ax.set_ylabel("Number of Ratings")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(figures_dir / "user_long_tail.png", dpi=300)

    plt.close(fig)

# Generate and save the rating distribution plot

def plot_rating_distribution(rating_counts, figures_dir):

    figures_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.bar(rating_counts.index, rating_counts.values)

    ax.set_title("Distribution of Movie Ratings")
    ax.set_xlabel("Rating")
    ax.set_ylabel("Number of Ratings")
    ax.set_xticks([1, 2, 3, 4, 5])

    fig.tight_layout()
    fig.savefig(figures_dir / "rating_distribution.png", dpi=300)

    plt.close(fig)

# Generate and save the monthly rating activity plot

def plot_monthly_activity(monthly_activity, figures_dir):

    figures_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(monthly_activity.index, monthly_activity.values)

    ax.set_title("Monthly Rating Activity")
    ax.set_xlabel("Month")
    ax.set_ylabel("Number of Ratings")
    ax.grid(alpha=0.3)

    fig.autofmt_xdate()

    fig.tight_layout()
    fig.savefig(figures_dir / "monthly_activity.png", dpi=300)

    plt.close(fig)

# Calculate the percentage of ratings contributed by the top 10% of entities

def top_activity_share(counts):

    n_top = math.ceil(len(counts) * 0.10)

    top_ratings = counts.nlargest(n_top).sum()

    total_ratings = counts.sum()

    return top_ratings / total_ratings * 100

# Check the dataset for missing values, duplicate ratings, and unknown IDs

def data_quality_checks(ratings, movies, users):

    missing_values = (
        ratings.isna().sum().sum()
        + movies.isna().sum().sum()
        + users.isna().sum().sum()
    )

    duplicate_ratings = ratings.duplicated(
        subset=["UserID", "MovieID"]
    ).sum()

    unknown_users = (
        ~ratings["UserID"].isin(users["UserID"])
    ).sum()

    unknown_movies = (
        ~ratings["MovieID"].isin(movies["MovieID"])
    ).sum()

    return {
        "missing_values": int(missing_values),
        "duplicate_ratings": int(duplicate_ratings),
        "unknown_users": int(unknown_users),
        "unknown_movies": int(unknown_movies)
    }