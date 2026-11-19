# Import pandas for working with DataFrames
import pandas as pd

# Import NumPy for numerical operations
import numpy as np

# Define a function to convert movie genres into numerical features
def encode_genres(movies: pd.DataFrame) -> pd.DataFrame:

    # Create one column for each genre, using 1 if the movie belongs to it and 0 otherwise
    genre_features = movies["Genres"].str.get_dummies(sep="|")

    # Convert the values to float32 for future mathematical calculations
    genre_features = genre_features.astype("float32")

    # Use MovieID as the index so each row can be identified by its movie ID
    genre_features.index = movies["MovieID"].to_numpy()

    # Give the index the name MovieID
    genre_features.index.name = "MovieID"

    # Return the numerical genre feature matrix
    return genre_features

# Convert movie release years into numerical features
def encode_release_year(movies: pd.DataFrame) -> pd.DataFrame:

    # Extract the four-digit release year from the end of each movie title
    release_year = movies["Title"].str.extract(r"\((\d{4})\)$", expand=False)

    # Convert the extracted year from text into a number
    release_year = pd.to_numeric(release_year, errors="coerce")

    # Group release years into five-year intervals
    year_bucket = ((release_year // 5) * 5).astype("Int64")

    # Create one numerical column for each five-year interval
    year_features = pd.get_dummies(year_bucket, prefix="Year", dtype="float32")

    # Use movie IDs as the row index
    year_features.index = movies["MovieID"].to_numpy()

    # Name the index MovieID
    year_features.index.name = "MovieID"

    # Return the movie release-year feature matrix
    return year_features

# Combine movie genres and release-year features into one matrix
def build_item_features(movies: pd.DataFrame, year_weight: float = 1.0) -> pd.DataFrame:

    # Convert movie genres into numerical columns
    genre_features = encode_genres(movies)

    # Convert release years into numerical columns
    year_features = encode_release_year(movies)

    # Control how strongly release year influences recommendations
    year_features = year_features * year_weight

    # Combine genre and year columns using their MovieID index
    item_features = pd.concat([genre_features, year_features], axis=1)

    # Return the complete movie feature matrix
    return item_features

# Convert user ratings into preference weights
def rating_weights(train: pd.DataFrame, weight_mode: str = "centered") -> np.ndarray:

    # Get the ratings as a NumPy array
    ratings = train["Rating"].to_numpy(dtype=np.float32)

    # Center ratings around the neutral value of 3
    if weight_mode == "centered":
        weights = ratings - 3.0

    # Give positive ratings a weight of 1 and others 0
    elif weight_mode == "positive_only":
        weights = (ratings >= 4).astype(np.float32)

    # Reject unsupported weighting methods
    else:
        raise ValueError("Unknown weight mode")

    # Return the calculated preference weights
    return weights

# Build a preference profile for every user
def build_user_profiles(
    train: pd.DataFrame,
    item_features: pd.DataFrame,
    weight_mode: str = "centered"
) -> pd.DataFrame:

    # Convert ratings into preference weights
    weights = rating_weights(train, weight_mode)

    # Get unique users and their numerical positions
    user_ids, user_positions = np.unique(
        train["UserID"].to_numpy(),
        return_inverse=True
    )

    # Get the feature vectors of movies rated by users
    movie_vectors = item_features.loc[
        train["MovieID"]
    ].to_numpy(dtype=np.float32)

    # Multiply each movie's features by its rating weight
    weighted_vectors = movie_vectors * weights[:, np.newaxis]

    # Create an empty profile for every user
    profiles = np.zeros(
        (len(user_ids), item_features.shape[1]),
        dtype=np.float32
    )

    # Add weighted movie features to the corresponding user
    np.add.at(profiles, user_positions, weighted_vectors)

    # Convert the resulting profiles into a DataFrame
    user_profiles = pd.DataFrame(
        profiles,
        index=user_ids,
        columns=item_features.columns
    )

    # Name the user index
    user_profiles.index.name = "UserID"

    # Return the user preference profiles
    return user_profiles

# Calculate cosine similarity between a user and all movie vectors
def cosine_similarity_scores(
    user_vector: np.ndarray,
    item_matrix: np.ndarray
) -> np.ndarray:

    # Calculate the magnitude of the user vector
    user_norm = np.linalg.norm(user_vector)

    # Calculate the magnitude of each movie vector
    item_norms = np.linalg.norm(item_matrix, axis=1)

    # Calculate the dot product between each movie and the user
    dot_products = item_matrix @ user_vector

    # Calculate the denominator for each similarity score
    denominators = user_norm * item_norms

    # Divide safely, returning 0 when a vector has zero magnitude
    scores = np.divide(
        dot_products,
        denominators,
        out=np.zeros_like(dot_products, dtype=np.float64),
        where=denominators > 0
    )

    # Return the similarity scores for all movies
    return scores

# Calculate recommendation scores for all movies for one user
def score_movies_for_user(
    user_id: int,
    user_profiles: pd.DataFrame,
    item_features: pd.DataFrame
) -> pd.Series:

    # Get the preference vector of the selected user
    user_vector = user_profiles.loc[user_id].to_numpy(dtype=np.float32)

    # Match the movie feature columns to the user profile columns
    aligned_features = item_features.reindex(columns=user_profiles.columns)

    # Convert movie features into a NumPy matrix
    item_matrix = aligned_features.to_numpy(dtype=np.float32)

    # Calculate cosine similarity between the user and all movies
    scores = cosine_similarity_scores(user_vector, item_matrix)

    # Convert scores into a Series indexed by movie IDs
    movie_scores = pd.Series(
        scores,
        index=item_features.index,
        name="Prediction"
    )

    # Return the recommendation scores
    return movie_scores

# Remove movies already rated by the selected user
def exclude_seen_movies(
    user_id: int,
    train: pd.DataFrame,
    movie_scores: pd.Series
) -> pd.Series:

    # Get movies the user has already rated in training
    seen_movies = train.loc[
        train["UserID"] == user_id, "MovieID"
    ]

    # Keep only movies that are not in the user's history
    unseen_scores = movie_scores[
        ~movie_scores.index.isin(seen_movies)
    ]

    # Return scores for unseen movies only
    return unseen_scores

# Generate movie prediction scores for all target users
def generate_prediction_scores(
    target_users,
    train: pd.DataFrame,
    item_features: pd.DataFrame,
    user_profiles: pd.DataFrame
) -> pd.DataFrame:

    # Create a list to collect predictions for every user
    predictions = []

    # Process each target user separately
    for user_id in target_users:

        # Skip users without a profile in the training data
        if user_id not in user_profiles.index:
            continue

        # Calculate similarity scores for all movies
        movie_scores = score_movies_for_user(
            user_id, user_profiles, item_features
        )

        # Remove movies already rated by this user
        unseen_scores = exclude_seen_movies(
            user_id, train, movie_scores
        )

        # Create a DataFrame with the required evaluation columns
        user_predictions = pd.DataFrame({
            "UserID": user_id,
            "MovieID": unseen_scores.index,
            "Prediction": unseen_scores.to_numpy()
        })

        # Add this user's predictions to the results
        predictions.append(user_predictions)

    # Handle the case where no predictions were generated
    if not predictions:
        return pd.DataFrame(
            columns=["UserID", "MovieID", "Prediction"]
        )

    # Combine all users' predictions into one DataFrame
    return pd.concat(predictions, ignore_index=True)

# Define the rating-weighting methods to compare
WEIGHT_MODES = ("centered", "positive_only")

# Define the release-year feature weights to compare
YEAR_WEIGHTS = (0.0, 0.5, 1.0, 2.0)
