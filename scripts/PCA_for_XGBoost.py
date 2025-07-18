import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
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

# Drop NAs
cols = [
    'AvgOrderValue',
    'AvgPromoDiscount',
    'AvgQuantity',
    'ShippingSimple',
    'WorkHours',
    'Holiday',
    'Summer',
    'ElectionSeason',
    'PresidentialElection'
]
df = customer_summary[cols].dropna().copy()

# Preprocessing: scale numerics, one-hot categoricals
numeric_features = ['AvgOrderValue', 'AvgPromoDiscount', 'AvgQuantity']
categorical_features = ['ShippingSimple', 'WorkHours', 'Holiday', 'Summer', 'ElectionSeason', 'PresidentialElection']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(drop='first'), categorical_features)
    ]
)

# PCA pipeline
pipeline = Pipeline(steps=[
    ('pre', preprocessor),
    ('pca', PCA(n_components=5))
])

X_pca = pipeline.fit_transform(df)

# Explained variance
pca = pipeline.named_steps['pca']
explained = pca.explained_variance_ratio_

print("Explained variance by component:")
for i, val in enumerate(explained, 1):
    print(f"PC{i}: {val:.3f}")

# Get the PCA component loadings (aka weights)
loadings = pd.DataFrame(
    pca.components_.T,
    columns=[f'PC{i+1}' for i in range(pca.n_components_)],
    index=pipeline.named_steps['pre'].get_feature_names_out()
)

# Display top contributing features for each component
for pc in loadings.columns:
    print(f"\nTop features for {pc}:")
    print(loadings[pc].sort_values(key=abs, ascending=False).head(5))


# Scree plot
plt.plot(range(1, len(explained) + 1), np.cumsum(explained), marker='o')
plt.title("Cumulative Explained Variance")
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Variance Explained")
plt.grid(True)
plt.tight_layout()
plt.show()
