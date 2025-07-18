import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
# from sklearn.utils import resample
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

order_df_retention = pd.read_feather("C:/Users/Graphicsland/Spyder/retention/outputs/order_df_retention.feather")
customer_summary = pd.read_feather("C:/Users/Graphicsland/Spyder/retention/outputs/customer_summary.feather")

keep_customers = customer_summary['CustomerId'][customer_summary['TotalOrders'] > 1]

order_df_retention = order_df_retention[order_df_retention['CustomerId'].isin(keep_customers)]
customer_summary = customer_summary[customer_summary['CustomerId'].isin(keep_customers)]

order_df_retention['OrderDate'] = pd.to_datetime(order_df_retention['OrderDate'])
order_df_retention = order_df_retention.sort_values(['CustomerId', 'OrderDate'])
grouped = order_df_retention.groupby('CustomerId')


order_df_retention['Rolling2_AvgPrice'] = grouped['OrderItemPriceTotal'].rolling(2, min_periods=1).mean().reset_index(drop=True)
# order_df_retention['Rolling3_AvgPrice'] = grouped['OrderItemPriceTotal'].rolling(3, min_periods=1).mean().reset_index(drop=True)

order_df_retention['Rolling2_AvgShipping'] = grouped['RushFeeAndShipPrice'].rolling(2, min_periods=1).mean().reset_index(drop=True)
# order_df_retention['Rolling3_AvgShipping'] = grouped['RushFeeAndShipPrice'].rolling(3, min_periods=1).mean().reset_index(drop=True)

order_df_retention['Rolling2_AvgDiscount'] = grouped['TotalDiscount'].rolling(2, min_periods=1).mean().reset_index(drop=True)
# order_df_retention['Rolling3_AvgDiscount'] = grouped['TotalDiscount'].rolling(3, min_periods=1).mean().reset_index(drop=True)


order_df_retention['DaysSincePrevOrder'] = grouped['OrderDate'].diff().dt.days
order_df_retention['Rolling2_Gap'] = grouped['DaysSincePrevOrder'].rolling(2, min_periods=1).mean().reset_index(drop=True)
# order_df_retention['Rolling3_Gap'] = grouped['DaysSincePrevOrder'].rolling(3, min_periods=1).mean().reset_index(drop=True)


order_df_retention['RollingStd_Price'] = grouped['OrderItemPriceTotal'].rolling(3, min_periods=2).std().reset_index(drop=True)
order_df_retention['RollingStd_Shipping'] = grouped['RushFeeAndShipPrice'].rolling(3, min_periods=2).std().reset_index(drop=True)


# Compute Z-scores
def zscore(current, rolling_mean):
    return (current - rolling_mean) / (rolling_mean.replace(0, 1))  # Avoid div by zero

# Z-scores based on 2-order rolling averages
order_df_retention['Z_Price'] = zscore(order_df_retention['OrderItemPriceTotal'], order_df_retention['Rolling2_AvgPrice'])
order_df_retention['Z_Shipping'] = zscore(order_df_retention['RushFeeAndShipPrice'], order_df_retention['Rolling2_AvgShipping'])
order_df_retention['Z_Discount'] = zscore(order_df_retention['TotalDiscount'], order_df_retention['Rolling2_AvgDiscount'])
order_df_retention['Z_Gap'] = zscore(order_df_retention['DaysSincePrevOrder'], order_df_retention['Rolling2_Gap'])


# Flagging if the current value is >1.5 std deviations from rolling 2 average (can adjust threshold)
order_df_retention['Flag_LowPrice'] = order_df_retention['Z_Price'] < -1
order_df_retention['Flag_HighPrice'] = order_df_retention['Z_Price'] < 1
order_df_retention['Flag_LowShipping'] = order_df_retention['Z_Shipping'] > -1
order_df_retention['Flag_HighShipping'] = order_df_retention['Z_Shipping'] > 1
order_df_retention['Flag_LowDiscount'] = order_df_retention['Z_Discount'] > -1
order_df_retention['Flag_HighDiscount'] = order_df_retention['Z_Discount'] > 1
order_df_retention['Flag_SlowingCadence'] = order_df_retention['Z_Gap'] > 1
order_df_retention['Flag_IncreasingCadence'] = order_df_retention['Z_Gap'] > -1


last_orders = order_df_retention.sort_values(['CustomerId', 'OrderDate']).groupby('CustomerId').tail(1)

zscore_flags = last_orders[[
    'CustomerId',
    'OrderNumber',
    'OrderDate',
    'Z_Price',
    'Z_Shipping',
    'Z_Discount',
    'Z_Gap',
    'Flag_LowPrice',
    'Flag_HighPrice',
    'Flag_LowShipping',
    'Flag_HighShipping',
    'Flag_LowDiscount',
    'Flag_HighDiscount',
    'Flag_SlowingCadence',
    'Flag_IncreasingCadence'
]]

zscore_flags = zscore_flags.merge(customer_summary[['CustomerId', 'LikelyChurned']])



# Feature columns (you can tweak or expand this)
feature_cols = [
    'Z_Price',
    'Z_Shipping',
    'Z_Discount',
    'Z_Gap',
    # 'Flag_LowPrice',
    # 'Flag_HighPrice',
    # 'Flag_LowShipping',
    # 'Flag_HighShipping',
    # 'Flag_LowDiscount',
    # 'Flag_HighDiscount',
    # 'Flag_SlowingCadence',
    # 'Flag_IncreasingCadence'
]

# Only use numerical (non-binary) features for PCA to avoid weird component blending
numeric_features = ['Z_Price', 'Z_Shipping', 'Z_Discount', 'Z_Gap']
X_numeric = zscore_flags[numeric_features].fillna(0)

# Scale the data before PCA
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_numeric)

# Apply PCA
pca = PCA(n_components=None)  # You can increase this if needed
X_pca = pca.fit_transform(X_scaled)

print(pca.explained_variance_ratio_)



X = zscore_flags[feature_cols]
y = zscore_flags['LikelyChurned']

scale = len(y[y == False]) / len(y[y == True])



X = X.fillna(0)


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', scale_pos_weight=scale)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(classification_report(y_test, y_pred))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

xgb_importances = model.feature_importances_
importance_df = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': xgb_importances
}).sort_values(by='Importance', ascending=False)

sns.barplot(data=importance_df, x='Importance', y='Feature')
plt.title("XGBoost Feature Importance")
plt.show()




from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)









from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

y_pred_logreg = model.predict(X_test)

print(classification_report(y_test, y_pred_logreg))

sns.heatmap(confusion_matrix(y_test, y_pred_logreg), annot=True, fmt='d', cmap='Blues')
plt.title("Logistic Regression Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()









