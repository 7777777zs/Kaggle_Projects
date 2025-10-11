# Optiver Realized Volatility Prediction

  This project reproduces and extends a Kaggle solution for the [Optiver Realized Volatility Prediction](https://
  www.kaggle.com/competitions/optiver-realized-volatility-prediction) competition. The goal is to predict next-day
  realized volatility for each stock/time interval by engineering rich features from the competition’s order book and
  trade data, then blending gradient boosting and neural network models.

  ## Repository Layout

  - `Optiver_Realized_Volatility_Prediction.ipynb` – end-to-end notebook covering feature engineering, model training,
  cross-validation, and submission generation.
  - `submission.csv` (created by the notebook) – final competition submission file.

  ## Prerequisites

  - Python 3.8+ with common data-science libraries (`pandas`, `numpy`, `scikit-learn`, `lightgbm`, `tensorflow`,
  `matplotlib`, `seaborn`, `joblib`).
  - Kaggle API credentials configured locally so the notebook can access `../input/optiver-realized-volatility-
  prediction/`.

  Install the core dependencies:

  ```bash
  pip install pandas numpy scikit-learn lightgbm tensorflow matplotlib seaborn joblib
  ```

  ## Getting the Data

  1. Sign in to Kaggle and accept the competition rules.
  2. Use the Kaggle CLI to download and unzip the dataset in the repository parent (so the notebook sees it in `../
  input/optiver-realized-volatility-prediction/`):

  ```bash
  kaggle competitions download -c optiver-realized-volatility-prediction
  unzip optiver-realized-volatility-prediction.zip -d ../input/optiver-realized-volatility-prediction
  ```

  ## Notebook Workflow

  1. **Imports & Utility Functions**
     Loads plotting and ML libraries and defines helper functions for weighted average prices (WAP), log returns,
  realized volatility, and aggregation utilities.

  2. **Data Loading**
     Reads `train.csv` / `test.csv`, builds a composite `row_id = stock_id-time_id`, and enumerates stock IDs for
  parallel feature extraction.

  3. **Book Feature Engineering**
     - Reads `book_[train|test].parquet` per stock.
     - Derives WAP variants, spreads, volume imbalance, and log-return series.
     - Aggregates statistics across multiple “seconds-in-bucket” cutoffs (0, 100, 200, 300, 400, 500).
     - Computes custom price/volume descriptors (median absolute deviation, energy, IQR, directional counts).

  4. **Trade Feature Engineering**
     - Reads `trade_[train|test].parquet` per stock.
     - Creates price log-return volatility, order size/amount summaries, unique second counts, and additional price/
  volume tensors similar to the book pipeline.
     - Merges book and trade features on `row_id`.

  5. **Global Aggregations**
     Adds stock-level and time-level aggregates (mean/std/min/max of volatility metrics) and engineered “size tau”
  statistics derived from order counts.

  6. **Cluster-Based Features**
     Clusters stocks using KMeans on target correlations (seven clusters). Aggregates each cluster’s features by time,
  pivots to wide format, and merges back onto train/test.

  7. **Preprocessing for Models**
     Replaces infinities, applies `QuantileTransformer` to numerical features, and retains `stock_id`, `time_id`, and
  `target` for downstream modeling.

  8. **LightGBM Models**
     - Defines multiple parameter sets (`params0`, `params2`).
     - Uses 5-fold KFold with RMSPE weighting (`1 / target²`).
     - Trains models, reports out-of-fold RMSPE, stores feature importances, and generates test predictions.

  9. **Custom Fold Generation**
     Implements a k-means++-inspired splitter on the pivoted `time_id × stock_id` target matrix to build time-wise folds
  used for neural nets.

  10. **Neural Network Models**
      - Creates a `swish`-activated dense network with a learned `stock_id` embedding.
      - Trains two 5-fold runs (different seeds) with MinMax scaling inside each fold and early stopping / LR reduction
  callbacks.
      - Produces two sets of OOF/test predictions.

  11. **Ensembling & Submission**
      Blends LightGBM and NN predictions with manually tuned weights (several historical blends preserved as comments).
  Final blend writes `submission.csv`.

  ## Running the Notebook

  1. Launch Jupyter and open `Optiver_Realized_Volatility_Prediction.ipynb`.
  2. Execute cells sequentially. Expect long runtimes for feature generation (per-stock parquet processing) and model
  training, especially the high-boost LightGBM run and neural nets.

  ## Results

  - LightGBM models achieve competitive RMSPE based on 5-fold validation.
  - Two neural networks provide complementary predictions that improve the blended score.
  - Final blend matches the tuned settings from the original notebook (`submission.csv` around public LB score 0.1971).

  ## Next Steps

  - Experiment with alternative blending weights or stacking meta-models.
  - Explore feature pruning to speed training or reduce overfitting.
  - Evaluate different validation schemes (e.g., time-based splits that reflect market regimes).

  ## Acknowledgements

  Built on community ideas from the Optiver Kaggle forums and relies on the official competition dataset. Many thanks to
  Optiver and Kaggle for releasing the challenge.
