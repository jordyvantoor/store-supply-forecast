# Store Supply Forecast

Weekly unit-sales forecasts per item and store for the three Texas stores (TX_1, TX_2, TX_3) in the Walmart M5 dataset, turned into stocking decisions and served in a Streamlit planning app.

> Work in progress. Business results, the live app link and a screenshot will be added here once the backtest is done.

## How to run

Requires Python 3.11+. The setup below uses Miniconda for Python itself; all packages come from `requirements.txt`.

```bash
conda create -p ./.venv python=3.11 -y
conda activate ./.venv
pip install -r requirements.txt

# macOS only: XGBoost needs the OpenMP runtime
conda install -p ./.venv llvm-openmp -y

python -m forecast.download   # M5 raw files -> data/raw/
python -m forecast.data       # -> data/processed/tx_weekly.parquet
pytest -q
```

## Data

This project uses the **M5 Forecasting** dataset: daily unit sales of Walmart products, with prices and a calendar of events and SNAP days. Credit goes to **Walmart**, which provided the data, the **M5 competition organizers** (University of Nicosia, MOFC), and **Kaggle**, which hosted the competition. Raw data is downloaded by a script and is not committed to this repo.

Note: M5 contains sales, not demand. There is no inventory data, so a week with zero sales may mean no demand or no stock.
