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
- NumPy 2.2.6
- Matplotlib 3.10.8
- Seaborn 0.13.2

## Setup

Install the required dependencies:

```bash
python -m pip install -r requirements.txt
```

## Running the EDA

From the project root directory, run:

```bash
python run_all.py --data-dir data/ml-1m
```
If the dataset is stored elsewhere, replace data/ml-1m with the actual dataset folder path.

The script performs data quality checks, calculates EDA statistics, and saves four visualizations in outputs/figures/.

The EDA notebook is available in 01_Data_Exploration.ipynb.

The remaining recommendation models and evaluation components will be integrated as the project progresses.
