import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.metrics import precision_recall_curve
import matplotlib.patches as mpatches

# mask_dict = {
#     'OrderContainsLabel': True,
# }

mask_dict = {
    'OrderContainsSticker': True,
}

# mask_dict = None


# Set working directory
os.chdir('C:\\Users\\Graphicsland\\Spyder\\retention')

### ---------------------------- Load Data ----------------------------
order_df_retention = pd.read_feather("outputs/order_df_retention.feather")
customer_summary = pd.read_feather("outputs/customer_summary_first_order.feather")


one_and_done_df = pd.read_feather("outputs/customer_summary.feather")
one_and_done_df = one_and_done_df.drop(columns=['AvgOrderItemTotal'])

# Get first order info
first_order_df = (
    order_df_retention
    .sort_values("ShippedDateTime")
    .drop_duplicates(subset="CustomerId", keep="first")
    [["CustomerId", "ShippedDateTime", 'HourOfDay', 'DayOfWeek', 'Month', 'DayOfMonth', 'PromoCodeDiscount',
      'ShippingType', 'OrderItemPriceTotal', 'AvgItemQuantity', 'HourCategory', 'WorkHours', 'Workday', 
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

customer_summary['HasPromoDiscount'] = customer_summary['AvgPromoDiscount'] < 0


if mask_dict is not None:
    for k, v in mask_dict.items():
        customer_summary = customer_summary[customer_summary[k] == v]



prediction_feature = 'OneAndDone'
xgb_features = [
    'OrderItemPriceTotal',
    'AvgItemsPerOrder',
    'HasPromoDiscount',
    'ShippingSimple',
    # 'Holiday',
    # 'OrderContainsLabel',
    'AvgRushFeeAndShipPrice',
    # 'WorkHours'
]

xgb_data = customer_summary[[prediction_feature] + xgb_features].dropna().copy()

# Encode categoricals
categorical_cols = ['ShippingSimple',
                    'HasPromoDiscount'
                    # 'Holiday', 
                    # 'OrderContainsLabel'
                    ]
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    xgb_data[col] = le.fit_transform(xgb_data[col].astype(str))
    label_encoders[col] = le

# Split data
X = xgb_data[xgb_features]
y = xgb_data[prediction_feature]
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# ------------------------------
# Handle class imbalance
# ------------------------------
neg, pos = np.bincount(y_train)
scale_pos_weight = neg / pos

# ------------------------------
# Train XGBoost
# ------------------------------
xgb = XGBClassifier(
    use_label_encoder=False,
    eval_metric='logloss',
    scale_pos_weight=scale_pos_weight,
    random_state=42,
)
xgb.fit(X_train, y_train)

# ------------------------------
# Evaluate with custom threshold
# ------------------------------
y_probs = xgb.predict_proba(X_test)[:, 1]

# Adjust threshold here
threshold = 0.425
y_pred_thresh = (y_probs > threshold).astype(int)

# ------------------------------
# Dynamic title helper
# ------------------------------
if mask_dict:
    filters = ", ".join([f"{k}={v}" for k, v in mask_dict.items()])
    title_suffix = f"\n(Filter: {filters})"
else:
    title_suffix = ""

# ------------------------------
# Classification Report & Confusion Matrix
# ------------------------------
conf_matrix = confusion_matrix(y_test, y_pred_thresh)

# Generate classification report as dict and string
report_dict = classification_report(y_test, y_pred_thresh, output_dict=True)
report_text = classification_report(y_test, y_pred_thresh)

# ------------------------------
# Plot confusion matrix + text report
# ------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw={'width_ratios': [1, 1.5]})

# Confusion matrix heatmap
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax1,
            xticklabels=['Not Churned', 'Churned'],
            yticklabels=['Not Churned', 'Churned'])
ax1.set_title(f"Confusion Matrix{title_suffix}")
ax1.set_xlabel("Predicted")
ax1.set_ylabel("Actual")

# Classification report as text
ax2.axis('off')
wrapped_text = f"Classification Report (threshold = {threshold}):\n\n{report_text}"
ax2.text(0, 1, wrapped_text, fontsize=11, va='top', family='monospace')

plt.tight_layout()
plt.savefig("outputs/classification_report_and_matrix.png", dpi=300)
plt.show()

# ------------------------------
# Feature Importance
# ------------------------------
importances = xgb.feature_importances_
sorted_idx = np.argsort(importances)[::-1]

plt.figure(figsize=(8, 6))
plt.barh([xgb_features[i] for i in sorted_idx], importances[sorted_idx])
plt.gca().invert_yaxis()
plt.title(f"XGBoost Feature Importances{title_suffix}")
plt.xlabel("Importance")
plt.tight_layout()
plt.show()

# ------------------------------
# Precision-Recall vs Threshold
# ------------------------------
prec, rec, thresholds = precision_recall_curve(y_test, y_probs)
plt.plot(thresholds, prec[:-1], label='Precision')
plt.plot(thresholds, rec[:-1], label='Recall')
plt.axvline(threshold, color='gray', linestyle='--')
plt.xlabel('Threshold')
plt.ylabel('Score')
plt.title(f'Precision and Recall vs Threshold{title_suffix}')
plt.legend()
plt.tight_layout()
plt.show()

# ------------------------------
# SHAP Summary Plot
# ------------------------------
explainer = shap.Explainer(xgb, X_train)
shap_values = explainer(X_test)

shap.summary_plot(shap_values, X_test, show=False)
plt.tight_layout()
plt.title(f"SHAP Summary Plot{title_suffix}")
plt.savefig("outputs/shap_summary_plot.png", dpi=300)
plt.show()

# ------------------------------
# SHAP Dependence Plot for Top Feature
# ------------------------------
# top_feature = X_train.columns[np.argsort(shap_values.values.mean(axis=0))[-1]]
# shap.dependence_plot(top_feature, shap_values.values, X_test, show=False)
# plt.tight_layout()
# plt.title(f"SHAP Dependence Plot for {top_feature}{title_suffix}")
# plt.savefig(f"outputs/shap_dependence_{top_feature}.png", dpi=300)
# plt.show()


### Manual SHAP plots

# Choose axes
shap.dependence_plot(
    "OrderItemPriceTotal",              # Feature for X-axis and SHAP values
    shap_values.values,            # SHAP values (from your explainer)
    X_test,                        # Original data (unscaled, encoded)
    interaction_index="ShippingSimple",  # Feature to color by
    show=False                     # Let you customize/save plot if needed
)
plt.tight_layout()
plt.show()

# Choose axes
shap.dependence_plot(
    "ShippingSimple",              # Feature for X-axis and SHAP values
    shap_values.values,            # SHAP values (from your explainer)
    X_test,                        # Original data (unscaled, encoded)
    interaction_index="OrderItemPriceTotal",  # Feature to color by
    show=False                     # Let you customize/save plot if needed
)
plt.tight_layout()
plt.show()

# Choose axes
shap.dependence_plot(
    "OrderItemPriceTotal",              # Feature for X-axis and SHAP values
    shap_values.values,            # SHAP values (from your explainer)
    X_test,                        # Original data (unscaled, encoded)
    interaction_index="AvgRushFeeAndShipPrice",  # Feature to color by
    show=False                     # Let you customize/save plot if needed
)
plt.tight_layout()
plt.show()

# Choose axes
shap.dependence_plot(
    "AvgRushFeeAndShipPrice",              # Feature for X-axis and SHAP values
    shap_values.values,            # SHAP values (from your explainer)
    X_test,                        # Original data (unscaled, encoded)
    interaction_index="OrderItemPriceTotal",  # Feature to color by
    show=False                     # Let you customize/save plot if needed
)
plt.tight_layout()
plt.show()



# Number line
# Extract relevant arrays
shap_vals = shap_values.values[:, X_test.columns.get_loc("ShippingSimple")]
shipping = X_test["ShippingSimple"].values
order_val = X_test["OrderItemPriceTotal"].values

# Classify each point into a color group
colors = []
for s, v in zip(shipping, order_val):
    if s == 0 and v < 50:
        colors.append("purple")
    elif s == 0 and v >= 50:
        colors.append("green")
    elif s == 1 and v < 75:
        colors.append("orange")
    else:
        colors.append("blue")

# Create scatterplot (number line)
plt.figure(figsize=(10, 1.5))
plt.scatter(shap_vals, [0]*len(shap_vals), c=colors, alpha=0.6, s=20)
plt.yticks([])
plt.xlabel("SHAP Value for ShippingSimple")
plt.title(f"SHAP Value Number Line by Shipping Type and Order Value{title_suffix})")
plt.axvline(0, color='gray', linestyle='--', linewidth=1)

# Optional legend
legend_elements = [
    mpatches.Patch(color='purple', label='Normal, < $50'),
    mpatches.Patch(color='green', label='Normal, ≥ $50'),
    mpatches.Patch(color='orange', label='Other, ≥ $75'),
    mpatches.Patch(color='blue', label='Other, < $75'),
    
]
plt.legend(handles=legend_elements, bbox_to_anchor=(1.02, 1), loc='upper left')
plt.tight_layout()
plt.show()