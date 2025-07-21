import pandas as pd
import numpy as np
import os
from datetime import timedelta
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import gower
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import time

start_time = time.time()

# ───────────────────────── SETUP ─────────────────────────
number_of_clusters = 5
cutoff_date = pd.Timestamp('2023-01-01')
mask_dict = {'FirstOrderContainsLabel': True}

os.chdir('C:\\Users\\Graphicsland\\Spyder\\retention')

customer_summary = pd.read_feather("outputs/customer_summary_first_order.feather")

order_df = pd.read_feather('../sharedData/raw_order_df.feather')
order_df = (
    order_df[(order_df['IsDeleted'] != 1) & (order_df['AffiliateId'] == 2)]
    .dropna(subset=['ShippedDateTime'])
    .copy()
)

item_df  = pd.read_feather('../sharedData/raw_order_item_df.feather')
item_df = (
    item_df[(item_df['IsDeleted'] != 1)]
    .copy()
)

ppo      = pd.read_feather('../sharedData/raw_product_product_option_df.feather')

# ───────────────────────── CLEAN ─────────────────────────
ppo['IsSticker'] = ppo['Name'].str.contains('icker', case=False, na=False) & ~ppo['Name'].str.contains('ample', case=False, na=False)
ppo['IsLabel']   = ppo['Name'].str.contains('abel', case=False, na=False) & ~ppo['Name'].str.contains('ample', case=False, na=False)
ppo['IsPouch']   = ppo['Name'].str.contains('ouch', case=False, na=False) & ~ppo['Name'].str.contains('ample', case=False, na=False)

item_df = item_df.merge(
    ppo[['Id', 'IsSticker', 'IsLabel', 'IsPouch']],
    left_on='ProductProductOptionId', right_on='Id', how='left'
)

first_orders = (
    order_df
    .sort_values('PlacedDateTime')
    .groupby('CustomerId', as_index=False)
    .first()
)

first_orders['ShippingType'] = np.select(
    [(first_orders['OrderItemPriceTotal'] < 75) & (first_orders['ShippingTotal'] == 0)],
    ['NormalShipping'],
    default='OtherShipping'
)

flags = (
    item_df.groupby('OrderNumber')[['IsSticker', 'IsLabel', 'IsPouch']]
    .any()
    .rename(columns={
        'IsSticker': 'OrderContainsSticker',
        'IsLabel': 'OrderContainsLabel',
        'IsPouch': 'OrderContainsPouch'
    })
)

first_orders = first_orders.merge(flags, on='OrderNumber', how='left')

first_order_flags = first_orders[['CustomerId', 'ShippingType', 'OrderContainsLabel', 'OrderContainsSticker', 'OrderContainsPouch']].copy()
first_order_flags.rename(columns={
    'OrderContainsLabel': 'FirstOrderContainsLabel',
    'OrderContainsSticker': 'FirstOrderContainsSticker',
    'OrderContainsPouch': 'FirstOrderContainsPouch'
}, inplace=True)

customer_summary = customer_summary.merge(first_order_flags, on='CustomerId', how='left')
customer_summary = customer_summary[customer_summary['FirstOrderDate'] >= cutoff_date]

# ───────────────────────── FILTER ─────────────────────────
for k, v in mask_dict.items():
    customer_summary = customer_summary[customer_summary[k] == v]

# ───────────────────────── SET FEATURES ─────────────────────────
cluster_features = [
    'AvgOrderItemTotal',
    'AvgItemsPerOrder',
    'HasPromoDiscount',
    'ShippingType',
    'AvgRushFeeAndShipPrice',
]
prediction_feature = 'OneAndDone'

# Boolean conversion for clustering
customer_summary['HasPromoDiscount'] = customer_summary['AvgPromoDiscount'] < 0

xgb_data = customer_summary[['CustomerId', prediction_feature] + cluster_features].dropna().copy()
xgb_data_copy = xgb_data.copy()

categorical_cols = ['HasPromoDiscount', 'ShippingType']
numeric_cols = list(set(cluster_features) - set(categorical_cols))

xgb_data[categorical_cols] = xgb_data[categorical_cols].astype(str)

scaler = MinMaxScaler()
xgb_data[numeric_cols] = scaler.fit_transform(xgb_data[numeric_cols])
features_for_clustering = xgb_data[numeric_cols + categorical_cols]

# Optional subsample
# max_rows = 5000
# if len(xgb_data) > max_rows:
#     sampled_idx = xgb_data.sample(n=max_rows, random_state=42).index
#     xgb_data = xgb_data.loc[sampled_idx].reset_index(drop=True)
#     features_for_clustering = features_for_clustering.loc[sampled_idx].reset_index(drop=True)

# ───────────────────────── HIERARCHICAL CLUSTERING ─────────────────────────
gower_dist_matrix = gower.gower_matrix(features_for_clustering)
Z = linkage(gower_dist_matrix, method='ward')
hc_labels = fcluster(Z, t=number_of_clusters, criterion='maxclust')

xgb_data['HC_Cluster'] = hc_labels

# Add readable types (booleans and categories)
xgb_data['ShippingType_Label'] = pd.Categorical(
    xgb_data['ShippingType'],
    categories=['Rushed', 'Expedited', 'NormalShipping', 'No-Ship']
)
xgb_data['HasPromoDiscount'] = xgb_data['HasPromoDiscount'].map({'True': True, 'False': False}).astype(bool)

# ───────────────────────── MERGE CLUSTERS BACK FOR ACTUAL VALUES ─────────────────────────
xgb_data_full = xgb_data_copy.merge(
    xgb_data[['CustomerId', 'HC_Cluster']],
    on='CustomerId', how='left'
)

xgb_data_full['ShippingType_Label'] = pd.Categorical(
    xgb_data_full['ShippingType'],
    categories=['Rushed', 'Expedited', 'NormalShipping', 'No-Ship']
)

# Pull from the original (non-mutated) customer_summary source, BEFORE it was stringified
xgb_data_full = xgb_data_full.drop(columns=['HasPromoDiscount'], errors='ignore')  # just in case
xgb_data_full = xgb_data_full.merge(
    customer_summary[['CustomerId', 'AvgPromoDiscount']],
    on='CustomerId',
    how='left'
)
xgb_data_full['HasPromoDiscount'] = xgb_data_full['AvgPromoDiscount'] < 0



# ───────────────────────── ACTUAL VALUE SUMMARY ─────────────────────────
cluster_summary_actual = xgb_data_full.groupby('HC_Cluster').agg(
    AvgItemsPerOrder=('AvgItemsPerOrder', 'mean'),
    AvgOrderItemTotal=('AvgOrderItemTotal', 'mean'),
    AvgRushFeeAndShipPrice=('AvgRushFeeAndShipPrice', 'mean'),
    PctUsedPromo=('HasPromoDiscount', 'mean'),
    PctNormalShipping=('ShippingType_Label', lambda x: (x == 'NormalShipping').mean()),
    PctOneAndDone=('OneAndDone', 'mean'),
    ClusterSize=('HC_Cluster', 'count')
).round(2)

cluster_summary_actual['ClusterValue'] = cluster_summary_actual['AvgOrderItemTotal'] * cluster_summary_actual['ClusterSize']
cols = list(cluster_summary_actual.columns)
cols.remove('ClusterValue')
insert_pos = cols.index('AvgOrderItemTotal') + 1
cols.insert(insert_pos, 'ClusterValue')
cluster_summary_actual = cluster_summary_actual[cols]
cluster_summary_actual['ClusterValue'] = cluster_summary_actual['ClusterValue'].apply(lambda x: f"${x:,.2f}")

# ───────────────────────── DENDROGRAM ─────────────────────────
plt.figure(figsize=(18, 8))
dendrogram(Z, truncate_mode='level', p=6, color_threshold=0.8 * max(Z[:, 2]))
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Sample index')
plt.ylabel('Distance')
plt.show()

# ───────────────────────── Decision Tree ─────────────────────────
from sklearn.tree import DecisionTreeClassifier, plot_tree
# from sklearn.preprocessing import OneHotEncoder

# Make a copy to avoid modifying the original
X = xgb_data_full[cluster_features].copy()

# One-hot encode 'ShippingType'
X = pd.get_dummies(X, columns=['ShippingType'], drop_first=True)


# Drop rows with missing HC_Cluster (i.e., rows that weren’t clustered)
X['HC_Cluster'] = xgb_data_full['HC_Cluster']
X = X.dropna(subset=['HC_Cluster'])

# Separate features and target again
y = X.pop('HC_Cluster')

# Train decision tree
tree_clf = DecisionTreeClassifier(max_depth=3, random_state=42)
tree_clf.fit(X, y)

# Plot tree
plt.figure(figsize=(18, 8))
plot_tree(
    tree_clf,
    feature_names=X.columns,
    class_names=[str(c) for c in tree_clf.classes_],
    filled=True,
    rounded=True
)
plt.title("Decision Tree Explaining HC_Cluster")
plt.show()



# ───────────────────────── DONE ─────────────────────────
print(cluster_summary_actual.to_string())
duration = timedelta(seconds=(time.time() - start_time))
print(f"Script completed in {duration}")
