import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from kmodes.kprototypes import KPrototypes
from sklearn.preprocessing import MinMaxScaler
import os

# Set working directory
os.chdir('C:\\Users\\Graphicsland\\Spyder\\retention')

### ---------------------------- Load Data ----------------------------
order_df_retention = pd.read_feather("outputs/order_df_retention.feather")
customer_summary = pd.read_feather("outputs/customer_summary_first_order.feather")
one_and_done_df = pd.read_feather("outputs/customer_summary.feather")

# Get first order info
first_order_df = (
    order_df_retention
    .sort_values("ShippedDateTime")
    .drop_duplicates(subset="CustomerId", keep="first")
    [["CustomerId", 'HourOfDay', 'DayOfWeek', 'Month', 'DayOfMonth', 'PromoCodeDiscount',
      'ShippingType', 'AvgItemQuantity', 'HourCategory', 'WorkHours', 'Workday', 
      'FirstWeek', 'LastWeek', 'Holiday', 'Summer', 'ElectionSeason', 
      'PresidentialElection', 'RushFeeAndShipPrice', 
      'OrderContainsSticker', 'OrderContainsLabel']])

# Ensure ID match
for df in [customer_summary, one_and_done_df, first_order_df]:
    df['CustomerId'] = df['CustomerId'].astype(str)

# Merge metadata into customer_summary
customer_summary = (
    customer_summary
    .merge(one_and_done_df[['CustomerId', 'OneAndDone']], on='CustomerId', how='left')
    .merge(first_order_df, on='CustomerId', how='left'))

### ---------------------------- Clustering Prep ----------------------------
prediction_feature = 'OneAndDone'
cluster_features = [
    'AvgOrderValue', 'AvgPromoDiscount', 'AvgItemsPerOrder', 'ShippingType',
    'WorkHours', 'OrderContainsLabel'
]

# Fix negatives
customer_summary['AvgPromoDiscount'] = customer_summary['AvgPromoDiscount'].abs()

# Drop NAs and isolate relevant features
features_raw = customer_summary[['CustomerId', prediction_feature] + cluster_features].dropna().reset_index(drop=True)

# 🔹 SAMPLE
sample_size = 100_000  # change as needed
features_raw = features_raw.sample(n=sample_size, random_state=42).reset_index(drop=True)

# Identify columns
categorical_cols = ['ShippingType', 'WorkHours', 'OrderContainsLabel']
numeric_cols = list(set(cluster_features) - set(categorical_cols))

# Scale numerics
scaler = MinMaxScaler()
features_scaled = features_raw.copy()
features_scaled[numeric_cols] = scaler.fit_transform(features_scaled[numeric_cols])
features_scaled[categorical_cols] = features_scaled[categorical_cols].astype(str)

# Build matrix
X_matrix = features_scaled[cluster_features].to_numpy()
cat_col_inds = [features_scaled[cluster_features].columns.get_loc(col) for col in categorical_cols]

# 🔹 Cluster (Sampled)
print("Clustering now...")
kproto = KPrototypes(n_clusters=4, random_state=42, init='Huang', verbose=2)
clusters = kproto.fit_predict(X_matrix, categorical=cat_col_inds)

# Assign cluster
features = features_raw[['CustomerId', prediction_feature]].copy()
features['Cluster'] = clusters

### ---------------------------- Summary Plots ----------------------------
cluster_churn = features.groupby('Cluster')[prediction_feature].value_counts(normalize=True).unstack().fillna(0)
print("Cluster churn:\n", cluster_churn)
print("\nCluster sizes:\n", features['Cluster'].value_counts().sort_index())

cluster_churn.plot(kind='bar', stacked=True)
plt.title("Churn Rate by Cluster")
plt.ylabel("Proportion")
plt.xlabel("Cluster")
plt.legend(title=prediction_feature, labels=['Not Churned', 'Churned'])
plt.tight_layout()
plt.show()

# Attach unscaled for binning
features['AvgOrderValue'] = features_raw['AvgOrderValue']
features['AvgItemsPerOrder'] = features_raw['AvgItemsPerOrder']

features['AvgOrderValueBin'] = pd.cut(
    features['AvgOrderValue'],
    bins=[0, 100, 500, 1000, 5000, np.inf],
    labels=['$0-$100', '$100-$500', "$500-$1,000", "$1,000-$5,000", "5,000+"]
)
features['AvgItemsBin'] = pd.cut(
    features['AvgItemsPerOrder'],
    bins=[0, 0.5, 1, 2, 3, 5, 10, np.inf],
    labels=['0–0.5', '0.5–1', '1–2', '2–3', '3–5', '5–10', '10+']
)

# Plots
for col, title in [('AvgOrderValueBin', 'AvgOrderValue'), ('AvgItemsBin', 'AvgItemsPerOrder')]:
    ct = pd.crosstab(features['Cluster'], features[col], normalize='index')
    ct.plot(kind='bar', stacked=True)
    plt.title(f'Distribution of {title} (binned) by Cluster')
    plt.xlabel('Cluster')
    plt.ylabel('Proportion')
    plt.legend(title=f'{title} (Binned)', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

# Categorical plots
for col in categorical_cols:
    ct = pd.crosstab(features['Cluster'], features_raw[col], normalize='index')
    ct.plot(kind='bar', stacked=True)
    plt.title(f'Distribution of {col} by Cluster')
    plt.ylabel('Proportion')
    plt.xlabel('Cluster')
    plt.legend(title=col, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()
