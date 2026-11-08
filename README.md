# Recommender Systems Project

Tampere University - DATA.ML.360

## Dataset
MovieLens 1M dataset from GroupLens.

## Download and Setup

Download the MovieLens 1M dataset from https://grouplens.org/datasets/movielens/1m/

Extract the downloaded ZIP file.

Create a folder named data in the project directory if it does not already exist.

Place the extracted ml-1m folder inside data.

The required files should be located at:

data/ml-1m/ratings.dat

data/ml-1m/movies.dat

data/ml-1m/users.dat

The original dataset is not included in this repository.

## Requirements

- Python 3.10.19
- pandas 2.3.3
- NumPy 1.26.4
- Matplotlib 3.10.8
- Seaborn 0.13.2
- scipy 1.15.3
- scikit-learn 1.7.2
- contourpy 1.3.2

The full dependency list is pinned in requirements.txt.

## Setup

Install the required dependencies:

```bash
python -m pip install -r requirements.txt
```

## Running the EDA and validation

From the project root directory, run:

```bash
python run_all.py --data-dir data/ml-1m
```
If the dataset is stored elsewhere, replace data/ml-1m with the actual dataset folder path.

The script performs data quality checks, calculates EDA statistics, and saves four visualizations in outputs/figures/.

The EDA notebook is available in 01_Data_Exploration.ipynb.

The same command also validates item-based kNN (Cosine, Pearson, and Jaccard similarities with different k values) and nine popularity variants (30 days, 90 days, or all history; counted ratings ≥3, ≥4, or ≥5). Hyperparameters are selected using validation NDCG@10.

## Final evaluation and saved results

After validation, refit on training + validation data and evaluate on the test set:

```bash
python run_all.py --data-dir data/ml-1m --evaluate-test
```

Final evaluation saves the selected settings in outputs/frozen_test_config.json and reuses this file if it already exists. Metrics are Precision@10, Recall@10, NDCG@10, and catalogue coverage; ratings ≥4 are considered relevant.

Saved results can be inspected without rerunning the time-consuming experiments:

- results.csv: final test metrics.
- outputs/knn_validation_results.csv: kNN validation comparison.
- outputs/popularity_validation_results.csv: popularity validation comparison.

## Temporal Data Split

The project uses a chronological split of the MovieLens 1M ratings based on their timestamps. Before splitting, the ratings are sorted from earliest to latest to preserve temporal order.

The dataset is divided into three subsets:

Training set (70%): 700,146 ratings.

Validation set (10%): 100,021 ratings.

Test set (20%): 200,042 ratings.

The split is implemented in run_all.py. The script reports the number of ratings and date range for each subset and checks that the ratings are ordered chronologically.

This split provides separate datasets for model training, validation, and final testing while respecting the temporal order of the ratings.

# EDA Updates

The exploratory data analysis (EDA) pipeline includes monthly rating activity analysis and visualization.
The monthly activity calculation and plotting were updated to group ratings by month and convert monthly periods to timestamps for plotting. The script saves the generated EDA figures in outputs/figures/.

