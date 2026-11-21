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
- scikit-surprise==1.1.5

The full dependency list is pinned in requirements.txt.

## Setup

Install the required dependencies:

```bash
python -m pip install -r requirements.txt
```

## Run the complete project

From the project root directory, run the full pipeline with one command:

```bash
python run_all.py --data-dir data/ml-1m
```
If the dataset is stored elsewhere, replace data/ml-1m with the actual dataset folder path.

The script performs data quality checks, calculates EDA statistics, and saves four visualizations in outputs/figures/.

The project implements and evaluates five recommendation approaches:

Item-based kNN: Uses Cosine, Pearson, and Jaccard similarities with different numbers of neighbours.

Most Popular: Evaluates popularity variants using different time windows (30 days, 90 days, or all history) and rating thresholds (ratings ≥3, ≥4, or ≥5).

User-based kNN: Uses user similarity with Cosine, Pearson, and Jaccard similarities.

Matrix Factorization (SVD): Uses the Surprise library to train and tune a matrix factorization model.

Content-based Filtering: Uses movie genres and release years to build movie features and user profiles.

Hyperparameters are selected using validation NDCG@10. The selected configurations are then frozen, models are refitted using training and validation data, and final performance is evaluated on the test set.

The evaluation metrics are Precision@10, Recall@10, NDCG@10, and catalogue coverage. Ratings ≥4 are considered relevant.


The EDA notebook is available in 01_Data_Exploration.ipynb.

## Optional: run individual stages

Run only EDA and training/validation, leaving the test set unevaluated:

```bash
python run_all.py --data-dir data/ml-1m --validation-only
```

After validation, run only the final refit and test evaluation using saved choices/results:

```bash
python run_all.py --data-dir data/ml-1m --test-only
```

These optional stage flags are mutually exclusive.

To test the operation of our code wee recommend running --test-only on outputs/frozen_test_config.json data that is already computed by us and uploaded on github. The reason for this is simple -- the validation stage can take 10-20 minutes to run. 

## Saved results

TThe complete run saves fresh validation-selected settings in outputs/frozen_test_config.json before test evaluation, replacing any previous frozen settings. The test-only command reuses this file if it exists; otherwise, it selects and freezes settings from the saved validation results.

Metrics are Precision@10, Recall@10, NDCG@10, and catalogue coverage; ratings ≥4 are considered relevant.

Saved results can be inspected without rerunning the time-consuming experiments:

- results.csv: Final test metrics.

- outputs/knn_validation_results.csv: Item-based kNN validation comparison.

- outputs/popularity_validation_results.csv: Popularity validation comparison.

- outputs/user_knn_validation_results.csv: User-based kNN validation comparison.

- outputs/content_based_validation_results.csv: Content-based validation comparison.

- outputs/MF_SVD_validation_results.csv: Matrix factorization validation comparison.

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

The EDA visualizations include:

movie_long_tail.png: Distribution of ratings across movies.

user_long_tail.png: Distribution of rating activity across users.

rating_distribution.png: Distribution of ratings from 1 to 5.

monthly_activity.png: Monthly rating activity over time.

Part 2: Additional Recommendation Models

Part 2 extends the initial recommendation experiments by implementing three additional recommendation approaches alongside the original item-based kNN and Most Popular baselines.

User-based kNN

A user-based collaborative filtering model was implemented using user similarity. Cosine, Pearson, and Jaccard similarity measures were evaluated with different neighbourhood sizes.

Matrix Factorization (SVD)

Matrix Factorization was implemented using the Surprise library. Different hyperparameter configurations were evaluated, including the number of latent factors, training epochs, learning rate, and regularization.

Content-based Filtering

A content-based recommendation model was implemented using movie genres and release years. Movie features are used to construct user profiles and generate recommendations based on content similarity.

A## Additional Experiments and Analysis

The project also includes additional experiments to investigate recommendation quality and model behaviour.

- **Statistical Analysis:** Paired permutation testing and bootstrap confidence intervals are used to compare recommendation performance.

- **Disaggregated Analysis:** Model performance is examined across different user activity groups and early versus late test periods.

- **Sparsity Ablation:** The effect of reducing training interactions is investigated using 100%, 50%, and 20% of the available training data.

- **Equal-budget Sensitivity Analysis:** Hyperparameter search sensitivity is examined under equal search budgets.

The corresponding results are saved in the outputs/ directory.

## Final Test Results

The final test evaluation in the project report compares all five recommendation approaches.

| Model | Precision@10 | Recall@10 | NDCG@10 | Catalogue Coverage |
|---|---:|---:|---:|---:|
| Item-based kNN | 0.04695 | 0.01213 | 0.04778 | 22.97% |
| Most Popular | 0.20996 | 0.05412 | 0.22644 | 3.32% |
| User-based kNN | 0.05377 | 0.01344 | 0.05537 | 39.17% |
| Matrix Factorization (SVD) | 0.12899 | 0.03494 | 0.14133 | 16.77% |
| Content-based Filtering | 0.02576 | 0.00685 | 0.02950 | 33.43% |

These results show that Most Popular achieved the highest ranking performance, while User-based kNN achieved the highest catalogue coverage among the evaluated models.

## Project Structure

```text
Recommender-Systems-Project/
├── README.md
├── requirements.txt
├── config.json
├── run_all.py
├── 01_Data_Exploration.ipynb
├── .gitignore
├── results.csv
│
├── src/
│   ├── data_loader.py
│   ├── eda.py
│   ├── services.py
│   ├── evaluation.py
│   ├── most_popular.py
│   ├── item_based_kNN.py
│   ├── user_based_kNN.py
│   ├── matrix_factorization.py
│   ├── content_based.py
│   ├── final_evaluation.py
│   ├── statistical_analysis.py
│   ├── disaggregated_analysis.py
│   ├── sparsity_ablation.py
│   └── equal_budget_sensitivity.py
│
├── outputs/
│   ├── frozen_test_config.json
│   ├── popularity_validation_results.csv
│   ├── knn_validation_results.csv
│   ├── user_knn_validation_results.csv
│   ├── content_based_validation_results.csv
│   ├── MF_SVD_validation_results.csv
│   ├── statistical_paired_comparison.csv
│   ├── statistical_per_user_ndcg.csv
│   ├── disaggregated_results.csv
│   ├── sparsity_ablation_results.csv
│   ├── equal_budget_sensitivity.csv
│   └── figures/
│       ├── movie_long_tail.png
│       ├── user_long_tail.png
│       ├── rating_distribution.png
│       └── monthly_activity.png
│
├── reports/
│   ├── Recommender_Systems_Report.docx
│   └── Results_Group4.xlsx
│
└── data/
    └── ml-1m/
        ├── ratings.dat
        ├── movies.dat
        └── users.dat
```

Project Report

The final project report is available in:

reports/Recommender_Systems_Report.docx

The official course results workbook is available in:

reports/Results_Group4.xlsx