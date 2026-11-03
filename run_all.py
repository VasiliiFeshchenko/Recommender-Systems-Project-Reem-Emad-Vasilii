# Import tools for paths, configuration, and project analysis

import argparse
from pathlib import Path

from src.data_loader import load_ratings, load_movies, load_users
from src import eda

import pandas as pd
import numpy as np

# Read the dataset location from the command line
parser = argparse.ArgumentParser()

parser.add_argument(
    "--data-dir",
    type=Path,
    default=Path("data/ml-1m")
)

args = parser.parse_args()

data_dir = args.data_dir
figures_dir = Path("outputs/figures")


# Load the MovieLens datasets
ratings = load_ratings(data_dir)
movies = load_movies(data_dir)
users = load_users(data_dir)
# Create a chronological 70% / 10% / 20% train-validation-test split
timestamps = ratings["Timestamp"].to_numpy(copy=True)
chronological_order = np.argsort(timestamps, kind="stable")
ratings_sorted = ratings.iloc[chronological_order].reset_index(drop=True)

n_ratings = len(ratings_sorted)
train_end = int(n_ratings * 0.70)
validation_end = int(n_ratings * 0.80)

train = ratings_sorted.iloc[:train_end].copy()
validation = ratings_sorted.iloc[train_end:validation_end].copy()
test = ratings_sorted.iloc[validation_end:].copy()

# Confirm chronological ordering
assert ratings_sorted["Timestamp"].is_monotonic_increasing

# Print split sizes and timestamp ranges in UTC
for split_name, split_data in [
    ("Train", train),
    ("Validation", validation),
    ("Test", test),
]:
    dates = pd.to_datetime(split_data["Timestamp"], unit="s", utc=True)
    print(f"\n{split_name}: {len(split_data):,} ratings")
    print(f"Date range: {dates.min()} to {dates.max()}")

# Check data quality before performing EDA calculations
quality_results = eda.data_quality_checks(ratings, movies, users)

print("Data quality checks:", quality_results)


# Calculate EDA statistics
overview = eda.dataset_overview(ratings, movies, users)

ratings_per_user, ratings_per_movie = eda.interaction_counts(ratings)

# Print descriptive statistics for user and movie activity

print("\nRatings per user:")
print(ratings_per_user.describe())

print("\nRatings per movie:")
print(ratings_per_movie.describe())

rating_counts = eda.rating_distribution(ratings)

monthly_activity = eda.monthly_rating_activity(ratings)

# Calculate and display temporal activity statistics

busiest_month = monthly_activity.idxmax()
highest_monthly_count = monthly_activity.max()

print("Busiest month:", busiest_month.strftime("%B %Y"))
print("Ratings in busiest month:", highest_monthly_count)

# Calculate the rating date range and percentage of ratings from 2000

rating_dates = pd.to_datetime(ratings["Timestamp"], unit="s", utc=True)

print("First rating:", rating_dates.min())
print("Last rating:", rating_dates.max())

ratings_in_2000 = monthly_activity.loc["2000"].sum()
percentage_2000 = ratings_in_2000 / len(ratings) * 100

print(f"Ratings from 2000: {percentage_2000:.2f}%")

# Print dataset statistics
print("Dataset overview:", overview)

print("Top 10% users:",
      f"{eda.top_activity_share(ratings_per_user):.2f}%")

print("Top 10% movies:",
      f"{eda.top_activity_share(ratings_per_movie):.2f}%")


# Generate and save all four EDA figures
eda.plot_movie_long_tail(ratings_per_movie, figures_dir)
eda.plot_user_long_tail(ratings_per_user, figures_dir)
eda.plot_rating_distribution(rating_counts, figures_dir)
eda.plot_monthly_activity(monthly_activity, figures_dir)

print("EDA figures saved successfully.")
