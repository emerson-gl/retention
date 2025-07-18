import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from kmodes.kprototypes import KPrototypes
from sklearn.preprocessing import MinMaxScaler
import os

number_of_clusters = 4

mask_dict = {
    'OrderContainsLabel': True,
}

# mask_dict = {
#     'OrderContainsSticker': True,
# }

# mask_dict = None

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
      'OrderContainsSticker', 'OrderContainsLabel']]
)

# Ensure ID match
for df in [customer_summary, one_and_done_df, first_order_df]:
    df['CustomerId'] = df['CustomerId'].astype(str)

# Merge metadata into customer_summary
customer_summary = (
    customer_summary
    .merge(one_and_done_df[['CustomerId', 'OneAndDone']], on='CustomerId', how='left')
    .merge(first_order_df, on='CustomerId', how='left')
)

customer_summary['ShippingSimple'] = customer_summary['ShippingType'].apply(
    lambda x: 'Normal' if x == 'NormalShipping' else 'Other'
)

# Clean discount
customer_summary['HasPromoDiscount'] = customer_summary['AvgPromoDiscount'] < 0

if mask_dict is not None:
    for k, v in mask_dict.items():
        customer_summary = customer_summary[customer_summary[k] == v]


### ---------------------------- Set Variables ----------------------------
prediction_feature = 'OneAndDone'
cluster_features = [
    'AvgOrderItemTotal',
    'AvgItemsPerOrder',
    'HasPromoDiscount',
    'ShippingSimple',
    'AvgRushFeeAndShipPrice',
]



### ---------------------------- Prepare Features ----------------------------
# Drop NAs and isolate relevant features
features_raw = customer_summary[['CustomerId', prediction_feature] + cluster_features].dropna().reset_index(drop=True)

# Identify categorical and numeric
categorical_cols = [
    'ShippingSimple',
    'HasPromoDiscount'
]
numeric_cols = list(set(cluster_features) - set(categorical_cols))

# Scale numeric columns
scaler = MinMaxScaler()
features_scaled = features_raw.copy()
features_scaled[numeric_cols] = scaler.fit_transform(features_scaled[numeric_cols])

# Convert categoricals to string
features_scaled[categorical_cols] = features_scaled[categorical_cols].astype(str)

# Prepare matrix and categorical indices
X_matrix = features_scaled[cluster_features].to_numpy()
cat_col_inds = [features_scaled[cluster_features].columns.get_loc(col) for col in categorical_cols]

### ---------------------------- Run Clustering ----------------------------
kproto = KPrototypes(n_clusters=number_of_clusters , random_state=42, init='Huang')
clusters = kproto.fit_predict(X_matrix, categorical=cat_col_inds)

# Final feature DataFrame
features = features_raw[['CustomerId', prediction_feature]].copy()
features = features.reset_index(drop=True)
features['Cluster'] = clusters

### ---------------------------- Summary Plots ----------------------------
# Cluster churn rates
cluster_churn = features.groupby('Cluster')[prediction_feature].value_counts(normalize=True).unstack().fillna(0)

# Build output name for feather
feature_string = "_".join(cluster_features).lower()
feather_name = f"outputs/{number_of_clusters}_clusters_{feature_string}_oneanddone.feather"
cluster_churn.reset_index().to_feather(feather_name)

# Build title suffix for filtered analysis
if mask_dict:
    filters = ", ".join([f"{k}={v}" for k, v in mask_dict.items()])
    title_suffix = f"\n(Filter: {filters})"
else:
    title_suffix = ""

# Cluster churn plot
cluster_churn.plot(kind='bar', stacked=True)
plt.title("Churn Rate by Cluster" + title_suffix)
plt.ylabel("Proportion")
plt.xlabel("Cluster")
plt.legend(title=prediction_feature, labels=['Not Churned', 'Churned'])
plt.tight_layout()
plt.savefig("outputs/cluster_churn_bar.png", dpi=300)
plt.show()

# Save cluster size plot
cluster_counts = features['Cluster'].value_counts().sort_index()
sns.barplot(x=cluster_counts.index, y=cluster_counts.values)
plt.title("Cluster Sizes" + title_suffix)
plt.xlabel("Cluster")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("outputs/cluster_sizes_bar.png", dpi=300)
plt.show()

# Binned feature distributions
features['AvgOrderValueBin'] = pd.cut(
    features_raw['AvgOrderItemTotal'],  # corrected from AvgOrderValue
    bins=[0, 100, 500, 1000, 5000, np.inf],
    labels=['$0-$100', '$100-$500', "$500-$1,000", "$1,000-$5,000", "5,000+"]
)
features['AvgItemsBin'] = pd.cut(
    features_raw['AvgItemsPerOrder'],
    bins=[0, 1, 2, 3, 5, 10, np.inf],
    labels=['1', '2', '3', '4–5', '6–10', '11+']
)

for col, title in [('AvgOrderValueBin', 'AvgOrderItemTotal'), ('AvgItemsBin', 'AvgItemsPerOrder')]:
    ct = pd.crosstab(features['Cluster'], features[col], normalize='index')
    ct.plot(kind='bar', stacked=True)
    plt.title(f'Distribution of {title} (binned) by Cluster' + title_suffix)
    plt.xlabel('Cluster')
    plt.ylabel('Proportion')
    plt.legend(title=f'{title} (Binned)', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f"outputs/distribution_{col.lower()}_by_cluster.png", dpi=300)
    plt.show()

# Categorical distributions
for col in categorical_cols:
    ct = pd.crosstab(features['Cluster'], features_raw[col], normalize='index')
    ct.plot(kind='bar', stacked=True)
    plt.title(f'Distribution of {col} by Cluster' + title_suffix)
    plt.ylabel('Proportion')
    plt.xlabel('Cluster')
    plt.legend(title=col, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f"outputs/distribution_{col.lower()}_by_cluster.png", dpi=300)
    plt.show()
