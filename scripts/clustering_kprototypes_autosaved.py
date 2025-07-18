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

### ---------------------------- Set Variables ----------------------------
prediction_feature = 'OneAndDone'
cluster_features = [
    'AvgOrderValue', 'AvgPromoDiscount', 'AvgItemsPerOrder', 'ShippingType',
    'WorkHours',  
    'OrderContainsLabel'
]

# Clean discount
customer_summary['AvgPromoDiscount'] = customer_summary['AvgPromoDiscount'].abs()

### ---------------------------- Prepare Features ----------------------------
# Drop NAs and isolate relevant features
features_raw = customer_summary[['CustomerId', prediction_feature] + cluster_features].dropna().reset_index(drop=True)

# Identify categorical and numeric
categorical_cols = [
    'ShippingType', 'WorkHours',
    'OrderContainsLabel'
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
kproto = KPrototypes(n_clusters=4, random_state=42, init='Huang')
clusters = kproto.fit_predict(X_matrix, categorical=cat_col_inds)

# Final feature DataFrame
features = features_raw[['CustomerId', prediction_feature]].copy()
features = features.reset_index(drop=True)
features['Cluster'] = clusters

### ---------------------------- Summary Plots ----------------------------
# Cluster churn rates
cluster_churn = features.groupby('Cluster')[prediction_feature].value_counts(normalize=True).unstack().fillna(0)
print("Cluster churn:")
print(cluster_churn)

# Cluster sizes
print("\nCluster sizes:")
print(features['Cluster'].value_counts().sort_index())

cluster_churn.plot(kind='bar', stacked=True)
plt.title("Churn Rate by Cluster")
plt.ylabel("Proportion")
plt.xlabel("Cluster")
plt.legend(title=prediction_feature, labels=['Not Churned', 'Churned'])
plt.tight_layout()
plt.show()

### ---------------------------- Feature Distributions ----------------------------
# Attach unscaled data for binning
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

# Plot binned features
for col, title in [('AvgOrderValueBin', 'AvgOrderValue'), ('AvgItemsBin', 'AvgItemsPerOrder')]:
    ct = pd.crosstab(features['Cluster'], features[col], normalize='index')
    ct.plot(kind='bar', stacked=True)
    plt.title(f'Distribution of {title} (binned) by Cluster')
    plt.xlabel('Cluster')
    plt.ylabel('Proportion')
    plt.legend(title=f'{title} (Binned)', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

# Plot categorical columns
for col in categorical_cols:
    ct = pd.crosstab(features['Cluster'], features_raw[col], normalize='index')
    ct.plot(kind='bar', stacked=True)
    plt.title(f'Distribution of {col} by Cluster')
    plt.ylabel('Proportion')
    plt.xlabel('Cluster')
    plt.legend(title=col, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()



for col in categorical_cols:
    ct = pd.crosstab(features['Cluster'], features[col], normalize='index')
    ct.plot(kind='bar', stacked=True)
    plt.title(f'Distribution of {col} by Cluster')
    plt.ylabel('Proportion')
    plt.xlabel('Cluster')
    plt.legend(title=col, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()
    
    



from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix


# ---------------------------
# XGBoost on OneAndDone
# ---------------------------

xgb_features = [
    'AvgOrderValue',
    'AvgPromoDiscount',
    'AvgQuantity',
    'ShippingType',
    'WorkHours',
    'Holiday',
    'Summer',
    'ElectionSeason',
    'PresidentialElection'
]

# Drop rows with missing features or labels
xgb_data = customer_summary[[prediction_feature] + xgb_features].dropna()

# Encode categorical variables
xgb_data = xgb_data.copy()
cat_features = ['ShippingType', 'WorkHours', 'Holiday', 'Summer', 'ElectionSeason', 'PresidentialElection']
for col in cat_features:
    xgb_data[col] = LabelEncoder().fit_transform(xgb_data[col].astype(str))

# Split X and y
X = xgb_data[xgb_features]
y = xgb_data[prediction_feature]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# Train model
xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
xgb.fit(X_train, y_train)

# Evaluate
y_pred = xgb.predict(X_test)
print(classification_report(y_test, y_pred))

# Confusion matrix
conf = confusion_matrix(y_test, y_pred)
sns.heatmap(conf, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Churned', 'Churned'], yticklabels=['Not Churned', 'Churned'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('XGBoost Confusion Matrix')
plt.tight_layout()
plt.show()

