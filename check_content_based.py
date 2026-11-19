# Import Path to locate the dataset folder
from pathlib import Path

# Import NumPy to check prediction values
import numpy as np

# Import existing data loading functions
from src.data_loader import load_movies, load_ratings

# Import our Content-Based model functions
from src.content_based import build_item_features, build_user_profiles, generate_prediction_scores


# Define the MovieLens dataset location
data_dir = Path("data/ml-1m")

# Load movies and ratings
movies = load_movies(data_dir)
ratings = load_ratings(data_dir)

# Sort ratings chronologically
ratings = ratings.sort_values("Timestamp", kind="stable")

# Select the first 70% of ratings for training
train = ratings.iloc[:int(len(ratings) * 0.70)]

# Build movie feature vectors
item_features = build_item_features(movies)

# Build user preference profiles
user_profiles = build_user_profiles(train, item_features)

# Select a user who exists in the training data
user_id = 1161

# Generate predictions for all unseen movies
predictions = generate_prediction_scores(
    [user_id], train, item_features, user_profiles
)

# Verify that all similarity scores are valid numbers
assert np.isfinite(predictions["Prediction"].to_numpy()).all()

# Find movies already rated by this user
seen_movies = set(train.loc[train["UserID"] == user_id, "MovieID"])

# Verify that previously rated movies were excluded
assert set(predictions["MovieID"]).isdisjoint(seen_movies)

# Sort movies by similarity and select the best 10
top10 = predictions.sort_values(
    ["Prediction", "MovieID"],
    ascending=[False, True]
).head(10)

# Add actual movie titles and genres
top10 = top10.merge(
    movies[["MovieID", "Title", "Genres"]],
    on="MovieID",
    how="left"
)

# Verify that exactly ten recommendations exist
assert len(top10) == 10

# Display how many movies the user has already rated
print("Movies already rated:", len(seen_movies))

# Display how many unseen movies received scores
print("Unseen movies scored:", len(predictions))

# Print the top ten recommendations with their titles
print("\nTOP 10 MOVIE RECOMMENDATIONS")
print(top10[["MovieID", "Title", "Prediction"]].to_string(index=False))