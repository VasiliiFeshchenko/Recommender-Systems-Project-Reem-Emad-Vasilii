from pathlib import Path

import numpy as np
import pandas as pd


def main():

    outputs_dir = Path("outputs")

    # Validation results from all five models
    validation_files = [
        ("Item-Based kNN", "knn_validation_results.csv"),
        ("Most Popular", "popularity_validation_results.csv"),
        ("Content-Based", "content_based_validation_results.csv"),
        ("User-Based kNN", "user_knn_validation_results.csv"),
        ("Matrix Factorization", "MF_SVD_validation_results.csv"),
    ]

    # Equal-budget experiment settings
    budget = 8
    repetitions = 1000
    rng = np.random.default_rng(42)

    results = []

    for model_name, filename in validation_files:

        df = pd.read_csv(outputs_dir / filename)

        scores = df["NDCG@10"].to_numpy(dtype=float)

        if len(scores) < budget:
            raise ValueError(
                f"{model_name} has fewer than {budget} configurations"
            )

        if not np.isfinite(scores).all():
            raise ValueError(f"Invalid validation scores for {model_name}")

        # Repeat random selection of 8 existing configurations
        best_scores = []

        for _ in range(repetitions):

            sampled_scores = rng.choice(
                scores,
                size=budget,
                replace=False
            )

            best_scores.append(sampled_scores.max())

        results.append({
            "Model": model_name,
            "Original budget": len(scores),
            "Full-search best NDCG": scores.max(),
            "Equal-budget median NDCG": np.median(best_scores),
            "Equal-budget 5th percentile": np.percentile(best_scores, 5),
            "Equal-budget 95th percentile": np.percentile(best_scores, 95),
        })

    results_df = pd.DataFrame(results)

    output_path = outputs_dir / "equal_budget_sensitivity.csv"

    # Verify reproducibility before replacing the existing results
    if output_path.exists():
        previous = pd.read_csv(output_path)

        if (
            list(previous.columns) != list(results_df.columns)
            or not previous["Model"].equals(results_df["Model"])
            or not np.allclose(
                previous.drop(columns="Model").to_numpy(dtype=float),
                results_df.drop(columns="Model").to_numpy(dtype=float),
                rtol=1e-12,
                atol=1e-12
            )
        ):
            raise ValueError(
                "Reproduced results differ from existing CSV. "
                "Existing file was NOT overwritten."
            )

        print("Reproducibility check PASSED.")

    results_df.to_csv(output_path, index=False)

    print("\nEQUAL-BUDGET SENSITIVITY RESULTS")
    print(results_df.to_string(index=False))

    print("\nSaved:", output_path)


if __name__ == "__main__":
    main()
