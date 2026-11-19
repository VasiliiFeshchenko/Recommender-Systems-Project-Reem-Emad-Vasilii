# Import tools for paths, configuration, and project analysis

import argparse
from pathlib import Path

from src.data_loader import load_ratings, load_movies, load_users
from src import eda
from src import item_based_kNN
from src import most_popular
from src.evaluation import evaluate_ranking
# Import our Content-Based recommendation model
from src import content_based

import pandas as pd
import numpy as np

# Read the dataset location from the command line
parser = argparse.ArgumentParser(
    description="Run EDA, training/validation, and final test evaluation by default.",
)

parser.add_argument(
    "--data-dir",
    type=Path,
    default=Path("data/ml-1m")
)

modes = parser.add_mutually_exclusive_group()
modes.add_argument(
    "--validation-only",
    action="store_true",
    help="Run EDA and training/validation without evaluating the test set.",
)
modes.add_argument(
    "--test-only",
    action="store_true",
    help="Freeze saved validation choices, refit on train+validation and evaluate test.",
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

if args.test_only:
    from src.final_evaluation import run_final_evaluation

    run_final_evaluation(train, validation, test, movies, Path("outputs"))
    raise SystemExit(0)

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

target_users = item_based_kNN.get_target_users(validation, train)
results_dir = Path("outputs")
results_dir.mkdir(parents=True, exist_ok=True)

#------------------------------------------------
# item-based knn

#rows are movies, columns are users. The matrix is mostly empty.
train_movie_user_matrix = item_based_kNN.build_movie_user_matrix(train)

# Compare three similarities using the same k grid and evaluation protocol.
similarity_methods = {
    "Cosine": item_based_kNN.compute_cosine_similarity,
    "Pearson": item_based_kNN.compute_pearson_similarity,
    "Jaccard": item_based_kNN.compute_jaccard_similarity,
}
similarity_notes = {
    "Cosine": "zero-filled training rating vectors",
    "Pearson": "shared raters only; minimum 2; undefined correlations set to zero",
    "Jaccard": "overlap of training rater sets; all observed ratings count",
}
k_values = item_based_kNN.knn_param_grid["model__n_neighbors"]
validation_results = []

for similarity_name, compute_similarity in similarity_methods.items():
    print(f"\nComputing {similarity_name} similarities...", flush=True)
    movie_similarities = compute_similarity(train_movie_user_matrix)

    for k in k_values:
        print(f"\nEvaluating {similarity_name}, k={k} on validation...", flush=True)
        predictions = item_based_kNN.generate_prediction_scores(
            target_users, train, train_movie_user_matrix, movie_similarities, k,
            catalogue_ids=movies["MovieID"], verbose=False,
        )
        metrics = evaluate_ranking(
            predictions, validation, train, target_users, movies["MovieID"],
            cutoff=10, relevance_threshold=4,
        )
        # Column names and order match the template's Part 1 - Baselines sheet.
        validation_results.append({
            "Model": "Item-based kNN",
            "Variant / settings": "Item-mean-centered; positive neighbors; full catalogue",
            "Similarity": similarity_name, "k": k,
            **metrics, "Seed": 0,
            "Notes": (
                "VALIDATION ONLY, not final test results; rating>=4 relevant; "
                "training history>=5; macro accuracy; users without relevant unseen ratings "
                "excluded from accuracy and coverage; ties by MovieID ascending; "
                "cold items use training global mean; no usable neighbors use item mean; "
                "deterministic algorithm (seed unused); "
                "select by highest validation NDCG@10, then smaller k, then similarity name; "
                + similarity_notes[similarity_name]
            ),
        })
        # Save completed iterations so results remain available if a later run stops.
        results = pd.DataFrame(validation_results)
        results.to_csv(results_dir / "knn_validation_results.csv", index=False)
        print(" | ".join(f"{name}: {value:.6f}" for name, value in metrics.items()), flush=True)
        del predictions

    del movie_similarities

best_result = results.sort_values(
    ["NDCG@10", "k", "Similarity"], ascending=[False, True, True]
).iloc[0]
best_k = int(best_result["k"])
best_similarity = best_result["Similarity"]
print("\nValidation comparison:")
print(results[["Similarity", "k", "Precision@10", "Recall@10", "NDCG@10", "Catalogue coverage"]].to_string(index=False))
print(f"\nBest configuration by validation NDCG@10: {best_similarity}, k={best_k}")
print("Results saved to outputs/knn_validation_results.csv. Test set remains untouched.")

# Most-popular baselines: 3 counting windows x 3 minimum counted ratings.
# Evaluation relevance stays >=4 for every variant, just as it does for kNN.
popularity_windows = most_popular.POPULARITY_WINDOWS
popularity_thresholds = most_popular.RATING_THRESHOLDS
popularity_results = []

for window_name, window_days in popularity_windows:
    for min_rating in popularity_thresholds:
        print(f"\nEvaluating popularity: {window_name}, counted ratings>={min_rating}...", flush=True)
        popularity = most_popular.compute_popularity(
            train, movies["MovieID"], window_days=window_days, min_rating=min_rating,
        )
        predictions = most_popular.generate_prediction_scores(target_users, train, popularity)
        metrics = evaluate_ranking(
            predictions, validation, train, target_users, movies["MovieID"],
            cutoff=10, relevance_threshold=4,
        )
        popularity_results.append({
            "Model": "Most popular",
            "Variant / settings": f"{window_name}; count ratings>={min_rating}",
            "Similarity": "", "k": "", **metrics, "Seed": 0,
            "Notes": (
                "VALIDATION ONLY, not final test results; evaluation relevance rating>=4; "
                "training history>=5; full catalogue; exclude all training-rated movies; "
                "macro accuracy; users without relevant unseen validation ratings excluded "
                "from accuracy and coverage; zero count for movies without qualifying ratings; "
                "ties by MovieID ascending; deterministic algorithm (seed unused); "
                f"window ends at training max timestamp={int(train['Timestamp'].max())}; "
                "recent windows include timestamps>=end-days*86400; "
                "compare by validation NDCG@10; ties keep window/threshold iteration order"
            ),
        })
        popularity_comparison = pd.DataFrame(popularity_results)
        popularity_comparison.to_csv(results_dir / "popularity_validation_results.csv", index=False)
        print(" | ".join(f"{name}: {value:.6f}" for name, value in metrics.items()), flush=True)
        del predictions

print("\nPopularity validation comparison:")
print(popularity_comparison[["Variant / settings", "Precision@10", "Recall@10", "NDCG@10", "Catalogue coverage"]].to_string(index=False))
best_popularity = popularity_comparison.sort_values(
    "NDCG@10", ascending=False, kind="stable"
).iloc[0]
print(f"\nBest popularity variant by validation NDCG@10: {best_popularity['Variant / settings']}")
print("Results saved to outputs/popularity_validation_results.csv. Test set remains untouched.")

# ------------------------------------------------
# Content-Based Recommender: Validation
# ------------------------------------------------

# Store the results of all eight configurations
content_results = []

# Try each rating-weighting method
for weight_mode in content_based.WEIGHT_MODES:

    # Try each release-year weight
    for year_weight in content_based.YEAR_WEIGHTS:

        # Show the configuration being evaluated
        print(
            f"\nContent-Based: {weight_mode}, year_weight={year_weight}",
            flush=True
        )

        # Build movie features using the selected year weight
        item_features = content_based.build_item_features(
            movies, year_weight=year_weight
        )

        # Build user profiles using training ratings only
        user_profiles = content_based.build_user_profiles(
            train, item_features, weight_mode=weight_mode
        )

        # Collect the top-ten predictions from each user batch
        top10_batches = []

        # Process 50 users at a time to limit memory usage
        for start in range(0, len(target_users), 50):

            # Select the current batch of users
            batch_users = target_users[start:start + 50]

            # Score all unseen movies for these users
            batch_predictions = content_based.generate_prediction_scores(
                batch_users, train, item_features, user_profiles
            )

            # Sort predictions from highest to lowest similarity
            batch_predictions = batch_predictions.sort_values(
                ["UserID", "Prediction", "MovieID"],
                ascending=[True, False, True]
            )

            # Keep the ten highest-scoring movies for each user
            batch_top10 = batch_predictions.groupby("UserID").head(10)

            # Save this batch's top-ten predictions
            top10_batches.append(batch_top10)

        # Combine the predictions from all user batches
        predictions = pd.concat(top10_batches, ignore_index=True)

        # Evaluate using the team's existing evaluation function
        metrics = evaluate_ranking(
            predictions,
            validation,
            train,
            target_users,
            movies["MovieID"],
            cutoff=10,
            relevance_threshold=4
        )

        # Record the configuration and its evaluation metrics
        content_results.append({
            "Model": "Content-Based",
            "Weight mode": weight_mode,
            "Year weight": year_weight,
            **metrics
        })

        # Save the results after each completed configuration
        pd.DataFrame(content_results).to_csv(
            results_dir / "content_based_validation_results.csv",
            index=False
        )

        # Display the current configuration's performance
        print(metrics, flush=True)

# Convert all configuration results into a DataFrame
content_results_df = pd.DataFrame(content_results)

# Select the configuration with the highest validation NDCG
best_content = content_results_df.sort_values(
    ["NDCG@10", "Weight mode", "Year weight"],
    ascending=[False, True, True]
).iloc[0]

# Display the complete validation comparison
print("\nCONTENT-BASED VALIDATION RESULTS")
print(content_results_df.to_string(index=False))

# Display the best configuration
print("\nBEST CONTENT-BASED CONFIGURATION")
print(best_content.to_string())

if not args.validation_only:
    from src.final_evaluation import run_final_evaluation

    print("\nValidation complete. Starting final test evaluation...", flush=True)
    run_final_evaluation(
        train, validation, test, movies, results_dir, refresh_config=True,
    )
