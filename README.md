# Retention Project

This repository contains a set of scripts and workflows for analyzing customer retention, churn risk, and purchase behavior at a custom printing company. The project aggregates customer- and order-level data, 
builds churn flags, and applies clustering and machine learning models to identify patterns and early-warning signals.

To be honest, this project was massively under construction at the time of my departure, so there are a number of extraneous/similar scripts, and the existing scripts were not yet generalized
to handle "one-and-done" analysis or churn of any type (read: it's a mess). The focus was initially on one-and-done which I thought might be an easier proving ground than all churn types.

## Repository Structure

### Data Inputs

Feather files exported from the production database and BigQuery (stored in `../sharedData`).

Important: Much of the analysis hinges on a feather called customer_summary (with and without suffixes like '_first_order') which is built in the script aggregate_customer_behavior.py. 
The script allows the user to select if a version of customer_summary with just a certain number of each customer's first or last orders selected, but all churn types are first assessed
by generating a customer_summary of all orders of all customers, which is then filtered and saved as 'churn_save.feather' in the project's outputs folder. In order to set the first_or_last
variable to 'first' or 'last', the script must first be run once with first_or_last = None.

I set up a folder called sharedData with no subfolders because I found myself using the same data across multiple projects. See the sharedData repository readme for more info.
A current snapshot (as of late July 2025) was uploaded here: "\\files.apps.gl\Teams\Product\Data\sharedData" 

### Outputs

Scripts write cleaned and aggregated data to `outputs/`:

* `order_df_retention.feather` — order-level features
* `customer_summary.feather` — customer-level aggregates
* `churn_save.feather` — churn pattern flags
* Model outputs (plots, reports, cluster assignments)

Plots were not set up to be automatically saved--I used the Spyder GUI to save multiple plots to a directory en masse.

### Scripts

#### 1. `aggregate_customer_behavior.py`

* Loads raw orders, items, and product option data.
* Cleans, filters, and enriches orders with features such as:

  * Time of day, day of week, month, holiday/summer/election flags
  * Shipping speed classification
  * Promo discount, rush fee, and shipping costs
  * Sticker/Label/Pouch indicators
* Aggregates to customer level:

  * Average order size, quantity, cadence
  * Percentage encodings for shipping, timing, and product mix
* Detects **seasonal patterns** (holidays, summers, elections) and cadence-based churn.
* Outputs:

  * `order_df_retention.feather`
  * `customer_summary.feather`
  * `churn_save.feather`
 
This is all data from our database. Time of day, day of week, etc. are all based on [Order].PlacedDateTime. For attribution analysis,
it might make sense to compute new temporal flags, but I did not have the chance to see if the PlacedDateTime differed significantly
from session start time as recorded by Google.

#### 2. `PCA.py` & `PCA_for_XGBoost.py`

Meant to reduce dimensionality of given model features. `PCA.py` also contains a rudimentary KMeans analysis.

#### 3. `XGBoost_oneanddone.py`

* Builds an **XGBoost classifier** to predict "OneAndDone" customers (churn after first order).
* Integrates **BigQuery session data** (traffic source, device, geo) into the feature set.
* Handles class imbalance with `scale_pos_weight`.
* Outputs:

  * Confusion matrix + classification report
  * Feature importance plots
  * Precision-recall vs threshold plot
  * SHAP summary plots for interpretability

#### 4. `clustering_kprototypes_oneanddone.py`

* Loads `customer_summary_first_order.feather`.
* Segments customers using **k-prototypes clustering** on:

  * Order size, item counts, promo usage, shipping type, and fees.
* Allows filtering (e.g., only label customers).
* Produces:

  * Cluster assignments with churn rates
  * Visuals: churn rate by cluster, cluster sizes, pairplot

#### 5. Additional scripts

There are a number of other scripts that may be of use, but those were the one getting most of my attention. `hierarchical.py` 
is extremely resource intensive even with subsampling, and didn't produce anything better than other models. `bigquery_crosstabs.py` is
nice for a quick visual of one-and-done churn behavior given attribution data.

## Workflow

1. **Run `aggregate_customer_behavior.py`** with first_or_last = None
   Generates the cleaned and aggregated order + customer-level datasets.
   Run again with first_or_last = 'first' as necessary for use by other scripts.

2. **Run modeling scripts:**

   * `PCA.py` for dimensionality-reduced numeric clustering.
   * `clustering_kprototypes_oneanddone.py` for categorical + numeric clustering.
   * `XGBoost_oneanddone.py` for churn classification and feature importance.

3. **Review outputs** in `outputs/` (Feather files + PNG charts).

---

## Dependencies

Core Python packages:

* `pandas`, `numpy`
* `scikit-learn`
* `xgboost`
* `shap`
* `matplotlib`, `seaborn`
* `kmodes` (for k-prototypes clustering)

Install with:

```bash
pip install pandas numpy scikit-learn xgboost shap matplotlib seaborn kmodes
```

---

## Next Steps

* Expand churn detection beyond first orders (multi-order cadence changes).
* Deploy early warning system to flag at-risk customers in production.
