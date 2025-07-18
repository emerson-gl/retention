import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
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
customer_summary['AvgPromoDiscount'] = customer_summary['AvgPromoDiscount'].abs()

# Feature selection
selected_features = [
    'TotalOrders',
    'AvgOrderValue',
    # 'MaxOrderValue',
    # 'MinOrderValue',
    # 'LifetimeValue',
    # 'AvgOrderItemTotal',
    # 'AvgPriceTotal',
    'AvgPromoDiscount',
    # 'AvgQuantity',
    # 'DaysSinceLastOrder'
]

# Drop missing
data = customer_summary[['LikelyChurned', 'CustomerId'] + selected_features].dropna()

# Clip for log-transform
X = data[selected_features].clip(lower=0.001)
X_log = np.log1p(X)

# Scale
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X_log)

# Target
y = data['LikelyChurned']

# Fit logistic regression
model = LogisticRegression(class_weight = 'balanced', max_iter=1000)
# model = LogisticRegression(class_weight={False: 12, True: 1}, max_iter=1000)
model.fit(X_scaled, y)

# Coefficients
coefs = pd.Series(model.coef_[0], index=selected_features)
odds_ratios = np.exp(coefs)

# Output
print("Logistic Regression Coefficients (Churn Odds Impact):")
print(coefs.sort_values(ascending=False))

print("\nOdds Ratios:")
print(odds_ratios.sort_values(ascending=False))

# Optional: evaluate model
y_pred = model.predict(X_scaled)
print("\nClassification Report:")
print(classification_report(y, y_pred))

# Visualize
sns.barplot(x=coefs.values, y=coefs.index)
plt.axvline(0, color='gray', linestyle='--')
plt.title("Feature Impact on Churn (Logistic Coefficients)")
plt.xlabel("Log-Odds Impact")
plt.tight_layout()
plt.show()


