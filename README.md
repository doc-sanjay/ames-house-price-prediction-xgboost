# Ames Housing Price Prediction | XGBoost Regression & SHAP

An end-to-end machine learning regression workflow predicting residential property sale prices on the Ames Housing dataset, achieving a Kaggle public leaderboard RMSLE of **0.13544**.

---

## 📌 Project Overview
* **Objective:** Predict residential home prices using 79 explanatory variables while addressing heavy target skewness, high missingness ratios, and multi-collinearity.
* **Evaluation Metric:** Root Mean Squared Logarithmic Error (RMSLE).
* **Final Model:** Regularized XGBoost Regressor (`objective='reg:squarederror'`).

---

## 🛠️ Data Preprocessing & Feature Engineering
* **Missing Value Differentiation:**
  * **Structural Absence:** Handled categorical amenities where `NaN` indicates the absence of a feature (`PoolQC`, `MiscFeature`, `Alley`, `Fence`, `GarageQual`, etc.) by imputing with `'None'`.
  * **Continuous Imputation:** Applied `KNNImputer(n_neighbors=5)` to estimate unrecorded measurements (`LotFrontage`, `GarageYrBlt`, `MasVnrArea`) based on multivariate proximity.
* **Feature Construction:**
  * Created `TotalHouseSF` by combining total basement area and above-grade total living area (including 1st and 2nd floors).
  * Engineered binary flags for rare/sparse features (`HasFence`, `HasAlley`, `Electrical_SBrkr`).
* **Categorical Encoding & Dimension Alignment:**
  * Mapped ordinal quality scales (`LotShape`, `PavedDrive`, `ExterCond`) to ordered numeric hierarchies.
  * Encoded remaining nominal features via one-hot encoding, ensuring exact feature alignment between train and test distributions without leakage.
* **Target Transformation:**
  * Applied `np.log1p(SalePrice)` prior to training to stabilize variance and directly align optimization with the RMSLE competition metric. Inverted during inference using `np.expm1`.

---

## 📊 Performance Benchmarks

| Metric | Local Validation Split | Kaggle Public Leaderboard |
| :--- | :--- | :--- |
| **RMSLE** | **0.1251** | **0.13544** |
| **R² Score** | **0.9071** | — |
| **MSE** | **0.0156** | — |

---

## 🔍 Model Explainability (SHAP)
Tree SHAP (`shap.TreeExplainer`) was utilized to extract global feature contributions and direction of impact:
* `OverallQual` and `TotalHouseSF` served as the dominant positive drivers of property valuation.
* Construction recency (`YearBuilt`) and neighborhood location provided secondary positive price shifts.

---

## 💻 Tech Stack
* **Language:** Python
* **Data Manipulation & Analysis:** pandas, NumPy
* **Machine Learning:** Scikit-Learn, XGBoost
* **Explainability & Visualization:** SHAP, Matplotlib, Seaborn
