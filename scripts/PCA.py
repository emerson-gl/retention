import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
order_df_retention = pd.read_feather("C:/Users/Graphicsland/Spyder/retention/outputs/order_df_retention.feather")
customer_summary = pd.read_feather("C:/Users/Graphicsland/Spyder/retention/outputs/customer_summary.feather")

# Cap days at 90
customer_summary['DaysSinceLastOrder'] = customer_summary['DaysSinceLastOrder'].clip(upper=90)
customer_summary['AvgDaysBetweenOrders'] = customer_summary['StdDaysBetweenOrders'].clip(upper=90)  # Optional

# Columns to exclude from clustering input
exclude_cols = [
    'CustomerId', 'LikelyChurned',
    'AvgRushFeeAndShipPrice', 'PctRushed', 'PctExpedited', 'PctNormalShipping', 'PctNo-Ship',
    'MeanDaysBetweenOrders'
]

# Keep numeric features only and drop excluded
X = customer_summary.drop(columns=exclude_cols, errors='ignore').select_dtypes(include=['number'])
X = X.dropna()

# Save associated CustomerIds and LikelyChurned for later merge/plot
customer_subset = customer_summary.loc

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA
pca = PCA(n_components=0.95)  # Retain ~95% variance
X_pca = pca.fit_transform(X_scaled)

print(f"Explained variance by components: {pca.explained_variance_ratio_}")
print(f"Number of components selected: {X_pca.shape[1]}")

# KMeans clustering
kmeans = KMeans(n_clusters=4, random_state=42)
customer_summary['Cluster'] = kmeans.fit_predict(X_pca)

# Optional: Churn breakdown
if 'LikelyChurned' in customer_summary.columns:
    churn_by_cluster = customer_summary.groupby('Cluster')['LikelyChurned'].value_counts(normalize=True).unstack().fillna(0)
    print(churn_by_cluster)

    # Visualize
    churn_by_cluster.plot(kind='bar', stacked=True)
    plt.title("Churn Rate by Cluster")
    plt.ylabel("Proportion")
    plt.xlabel("Cluster")
    plt.legend(title='LikelyChurned', labels=['Not Churned', 'Churned'])
    plt.tight_layout()
    plt.show()
