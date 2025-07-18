import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import RobustScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
order_df_retention = pd.read_feather("C:/Users/Graphicsland/Spyder/retention/outputs/order_df_retention.feather")
customer_summary = pd.read_feather("C:/Users/Graphicsland/Spyder/retention/outputs/customer_summary.feather")
prediction_feature = 'LikelyChurned'

day_cap = 90

# Cap days at value
customer_summary['DaysSinceLastOrder'] = customer_summary['DaysSinceLastOrder'].clip(upper=day_cap)
customer_summary['AvgDaysBetweenOrders'] = customer_summary['StdDaysBetweenOrders'].clip(upper=day_cap)  # Optional



# customer_summary = customer_summary[customer_summary['LikelyChurned'] == False]

# Fix negative discount values
customer_summary['AvgPromoDiscount'] = customer_summary['AvgPromoDiscount'].abs()

# Select relevant features
cluster_features = [
    'TotalOrders',
    'AvgOrderValue',
    # 'MaxOrderValue',
    # 'AvgItemsPerOrder',
    'AvgPromoDiscount',
    'PctNormalShipping',
    # 'PctOrdersWithSticker',
    # 'PctOrdersWithLabel'
]


# Prepare and clean data
features = customer_summary[[prediction_feature, 'CustomerId'] + cluster_features].copy()
features = features.dropna(subset=cluster_features)




# Clip all values to avoid log of negatives
clipped = features[cluster_features].clip(lower=0)

# Remove rows where all values are zero (optional safeguard)
clipped = clipped[(clipped > 0).any(axis=1)]

# Log-transform
log_transformed = np.log1p(clipped)

# Drop rows with any remaining NaNs just in case
log_transformed = log_transformed.dropna()

# Re-align features
features = features.loc[log_transformed.index]

# Scale
scaler = RobustScaler()
X_scaled = scaler.fit_transform(log_transformed)

# Cluster
kmeans = KMeans(n_clusters=3, random_state=42)
features['Cluster'] = kmeans.fit_predict(X_scaled)

# Analyze churn distribution
if prediction_feature in features.columns:
    cluster_churn = features.groupby('Cluster')[prediction_feature].value_counts(normalize=True).unstack().fillna(0)
    print(cluster_churn)

    cluster_churn.plot(kind='bar', stacked=True)
    plt.title("Churn Rate by Cluster")
    plt.ylabel("Proportion")
    plt.xlabel("Cluster")
    plt.legend(title=prediction_feature, labels=['Not Churned', 'Churned'])
    plt.tight_layout()
    plt.show()

# Cluster feature averages (normalized for comparison)
cluster_means = features.groupby('Cluster')[cluster_features].mean()
normalized_means = (cluster_means.T - cluster_means.T.mean(axis=1).values.reshape(-1,1)) / cluster_means.T.std(axis=1).values.reshape(-1,1)

normalized_means.plot(kind='bar', figsize=(12, 6))
plt.title('Normalized Cluster Feature Averages')
plt.ylabel('Z-score')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


print(features['Cluster'].value_counts(normalize=True))


