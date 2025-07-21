import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from kmodes.kprototypes import KPrototypes
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import os

# ───────────────────────── SETUP ─────────────────────────
cutoff_date = pd.Timestamp('2023-01-01')
number_of_clusters = 4
mask_dict = {'FirstOrderContainsSticker': True}  # or set to None

os.chdir('C:\\Users\\Graphicsland\\Spyder\\retention')

# ───────────────────────── LOAD ─────────────────────────
customer_summary = pd.read_feather("outputs/customer_summary_first_order.feather")
customer_summary['HasPromoDiscount'] = customer_summary['AvgPromoDiscount'] < 0


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
ppo['IsSticker'] = ppo['Name'].str.contains('icker',case=False,na=False) & ~ppo['Name'].str.contains('ample',case=False,na=False)
ppo['IsLabel']   = ppo['Name'].str.contains('abel' ,case=False,na=False) & ~ppo['Name'].str.contains('ample',case=False,na=False)
ppo['IsPouch']   = ppo['Name'].str.contains('ouch' ,case=False,na=False) & ~ppo['Name'].str.contains('ample',case=False,na=False)

item_df = item_df.merge(ppo[['Id','IsSticker','IsLabel', 'IsPouch']],
                        left_on='ProductProductOptionId', right_on='Id', how='left')


first_orders = (
    order_df
    .sort_values('PlacedDateTime')
    .groupby('CustomerId', as_index=False)
    .first()
)

first_orders['ShippingType'] = np.select(
    [(first_orders['OrderItemPriceTotal']<75)&(first_orders['ShippingTotal']==0)],
    ['NormalShipping'],
    default='OtherShipping'
)

flags = (
    item_df.groupby('OrderNumber')[['IsSticker','IsLabel', 'IsPouch']]
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
if mask_dict:
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

xgb_data = customer_summary[['CustomerId', prediction_feature] + cluster_features].dropna().copy()

# ───────────────────────── ENCODE + SCALE ─────────────────────────
categorical_cols = ['HasPromoDiscount', 'ShippingType']
numeric_cols = list(set(cluster_features) - set(categorical_cols))

# Encode categoricals as strings (required by k-prototypes)
xgb_data[categorical_cols] = xgb_data[categorical_cols].astype(str)

# Scale numeric features
scaler = MinMaxScaler()
xgb_data[numeric_cols] = scaler.fit_transform(xgb_data[numeric_cols])

# Prepare matrix and categorical indices
X_matrix = xgb_data[cluster_features].to_numpy()
cat_col_inds = [xgb_data[cluster_features].columns.get_loc(col) for col in categorical_cols]

# ───────────────────────── CLUSTER ─────────────────────────
kproto = KPrototypes(n_clusters=number_of_clusters, init='Huang', random_state=42)
clusters = kproto.fit_predict(X_matrix, categorical=cat_col_inds)

# Final DataFrame
features = xgb_data[['CustomerId', prediction_feature]].copy()
features['Cluster'] = clusters

# ───────────────────────── PLOTS ─────────────────────────
title_suffix = f"\n(Filter: {', '.join([f'{k}={v}' for k, v in mask_dict.items()])})" if mask_dict else ""

# Churn rate per cluster
churn_rate = features.groupby('Cluster')[prediction_feature].value_counts(normalize=True).unstack().fillna(0)
churn_rate.plot(kind='bar', stacked=True)
plt.title("Churn Rate by Cluster" + title_suffix)
plt.ylabel("Proportion")
plt.xlabel("Cluster")
plt.legend(title=prediction_feature, labels=['Not Churned', 'Churned'])
plt.tight_layout()
plt.savefig("outputs/cluster_churn_bar.png", dpi=300)
plt.show()

# Cluster sizes
sns.barplot(x=features['Cluster'].value_counts().index,
            y=features['Cluster'].value_counts().values)
plt.title("Cluster Sizes" + title_suffix)
plt.xlabel("Cluster")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("outputs/cluster_sizes_bar.png", dpi=300)
plt.show()



xgb_plot_data = xgb_data.copy()

for col in categorical_cols:
    if col == 'ShippingType':
        expected_shipping_types = ['Rushed', 'Expedited', 'NormalShipping', 'No-Ship', 'Missing']
        xgb_plot_data[col] = pd.Categorical(
            xgb_plot_data[col],
            categories=expected_shipping_types
        ).fillna('Missing')
        xgb_plot_data[col] = xgb_plot_data[col].cat.codes
    else:
        # Convert 'True'/'False' strings or actual bools to 1/0
        xgb_plot_data[col] = xgb_plot_data[col].map({True: 1, False: 0, 'True': 1, 'False': 0})


xgb_plot_data['Cluster'] = features['Cluster'].values

sns.pairplot(
    xgb_plot_data[numeric_cols + categorical_cols + ['Cluster']],
    hue='Cluster',
    palette='tab10',
    corner=True,
    plot_kws={'alpha': 0.4, 's': 15}
)
plt.suptitle("Corner Plot of Clustered Customers", fontsize=16)
plt.tight_layout()
plt.savefig("outputs/corner_plot_clusters.png", dpi=300)
plt.show()
