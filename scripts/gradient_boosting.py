import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load data
customer_summary = pd.read_feather("C:/Users/Graphicsland/Spyder/retention/outputs/customer_summary.feather")

# Cap outliers
day_cap = 90
customer_summary['DaysSinceLastOrder'] = customer_summary['DaysSinceLastOrder'].clip(upper=day_cap)
customer_summary['MeanDaysBetweenOrders'] = customer_summary['MeanDaysBetweenOrders'].clip(upper=day_cap)
customer_summary['AvgDiscount'] = customer_summary['AvgDiscount'].abs()

# Feature selection
selected_features = [
    'TotalOrders',
    'AvgOrderValue',
    'AvgDiscount',
    # 'AvgQuantity',
    # 'LifetimeValue',
    # 'DaysSinceLastOrder',
    # 'MeanDaysBetweenOrders'
]

# Drop missing
data = customer_summary[['LikelyChurned', 'CustomerId'] + selected_features].dropna()

# Clip for log-transform to avoid negatives/zero
X = data[selected_features].clip(lower=0.001)
X_log = np.log1p(X)

# Scale features
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X_log)

# Target
y = data['LikelyChurned']

# Split (optional, if you want a train/test split)
# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Fit gradient boosting classifier
gb_model = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)
gb_model.fit(X_scaled, y)

# Predict
y_pred = gb_model.predict(X_scaled)

# Evaluate model
print("\nClassification Report:")
print(classification_report(y, y_pred))

# Confusion matrix
conf_matrix = confusion_matrix(y, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Churned', 'Churned'], yticklabels=['Not Churned', 'Churned'])
plt.title("Confusion Matrix")
plt.ylabel("True label")
plt.xlabel("Predicted label")
plt.tight_layout()
plt.show()

# Feature importances
feature_importances = pd.Series(gb_model.feature_importances_, index=selected_features).sort_values(ascending=False)
print("\nFeature Importances:")
print(feature_importances)

# Plot feature importances
sns.barplot(x=feature_importances.values, y=feature_importances.index)
plt.title("Feature Importances (Gradient Boosting)")
plt.xlabel("Importance")
plt.tight_layout()
plt.show()
