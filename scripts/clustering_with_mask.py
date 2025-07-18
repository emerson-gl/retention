import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import RobustScaler
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# Load data
# -----------------------------
order_df_retention = pd.read_feather("C:/Users/Graphicsland/Spyder/retention/outputs/order_df_retention.feather")
customer_summary = pd.read_feather("C:/Users/Graphicsland/Spyder/retention/outputs/customer_summary.feather")

# -----------------------------
# Initial prep
# -----------------------------
day_cap = 90
customer_summary['DaysSinceLastOrder'] = customer_summary['DaysSinceLastOrder'].clip(upper=day_cap)
customer_summary['AvgDaysBetweenOrders'] = customer_summary['StdDaysBetweenOrders'].clip(upper=day_cap)

# Make AvgDiscount positive
customer_summary['AvgDiscount'] = customer_summary['AvgDiscount'].abs()

# -----------------------------
# Feature selection
# -----------------------------
cluster_features = [
    'TotalOrders',
    'AvgOrderValue',
    'AvgDiscount'
]

# -----------------------------
# Create base feature table
# -----------------------------
base_features = customer_summary[['LikelyChurned', 'CustomerId'] + cluster_features].copy()
base_features = base_features.dropna(subset=cluster_features)

# -----------------------------
# Transform full population
# -----------------------------
# Clip at zero, log1p transform, then robust scale
clipped_all = base_features[cluster_features].clip(lower=0)
clipped_all = clipped_all[(clipped_all > 0).any(axis=1)]
log_all = np.log1p(clipped_all)
log_all = log_all.dropna()

# Finalize the base features subset to match transformed rows
base_features = base_features.loc[log_all.index]

# Fit scaler on full dataset
scaler = RobustScaler()
X_all_scaled = scaler.fit_transform(log_all)
X_all_scaled_df = pd.DataFrame(X_all_scaled, columns=cluster_features, index=log_all.index)

# -----------------------------
# Subset to only non-churned (or change this condition)
# -----------------------------
subset_mask = base_features['LikelyChurned'] == False
X_subset = X_all_scaled_df[subset_mask]
features_subset = base_features[subset_mask].copy()

# -----------------------------
# Cluster on subset only
# -----------------------------
kmeans = KMeans(n_clusters=3, random_state=42)
features_subset['Cluster'] = kmeans.fit_predict(X_subset)

# -----------------------------
# Reattach full-pop z-scores
# -----------------------------
features_subset[cluster_features] = X_subset  # Already scaled to full-pop distribution



# -----------------------------
# Plot normalized cluster feature means (z-score relative to all)
# -----------------------------
cluster_means = features_subset.groupby('Cluster')[cluster_features].mean()
normalized_means = (cluster_means.T - X_all_scaled_df[cluster_features].mean().values.reshape(-1, 1)) / X_all_scaled_df[cluster_features].std().values.reshape(-1, 1)

normalized_means.plot(kind='bar', figsize=(12, 6))
plt.title('Normalized Cluster Feature Averages (Z-scores from full population)')
plt.ylabel('Z-score')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# -----------------------------
# Cluster distribution
# -----------------------------
print(features_subset['Cluster'].value_counts(normalize=True))
