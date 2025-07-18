import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import matplotlib.patches as mpatches


mask_feature = 'FirstOrderContainsLabel'
mask_dict = {mask_feature: True}

mask_dict = None


os.chdir('C:\\Users\\Graphicsland\\Spyder\\retention')

customer_summary = pd.read_feather("outputs/customer_summary_first_order.feather")
customer_summary['HasPromoDiscount'] = customer_summary['AvgPromoDiscount'] < 0

# ───────────────────────── 1  LOAD + CLEAN ─────────────────────────
order_df = pd.read_feather('../sharedData/raw_order_df.feather')
item_df  = pd.read_feather('../sharedData/raw_order_item_df.feather')
ppo      = pd.read_feather('../sharedData/raw_product_product_option_df.feather')

ppo['IsSticker'] = ppo['Name'].str.contains('icker',case=False,na=False) & ~ppo['Name'].str.contains('ample',case=False,na=False)
ppo['IsLabel']   = ppo['Name'].str.contains('abel' ,case=False,na=False) & ~ppo['Name'].str.contains('ample',case=False,na=False)
ppo['IsPouch']   = ppo['Name'].str.contains('ouch' ,case=False,na=False) & ~ppo['Name'].str.contains('ample',case=False,na=False)
item_df          = item_df.merge(ppo[['Id','IsSticker','IsLabel', 'IsPouch']],
                                 left_on='ProductProductOptionId', right_on='Id', how='left')

order_df = (
    order_df
    [ (order_df['IsDeleted'] != 1) & (order_df['AffiliateId'] == 2) ]
    .dropna(subset=['ShippedDateTime'])
    .copy()
)

first_orders = (
    order_df
    .sort_values('PlacedDateTime')
    .groupby('CustomerId', as_index=False)
    .first()  # or .head(1) if you want more flexibility
)

# shipping type
first_orders['ShippingType'] = np.select(
    [(first_orders['OrderItemPriceTotal']<75)&(first_orders['ShippingTotal']==0)],
    ['NormalShipping'],
    default='OtherShipping')


# basic per-order aggregates
counts   = item_df.groupby('OrderNumber').size().rename('ItemCount')
avg_qty  = item_df.groupby('OrderNumber')['Quantity'].mean().rename('AvgItemQuantity')
flags    = (item_df.groupby('OrderNumber')[['IsSticker','IsLabel', 'IsPouch']].any()
            .rename(columns={'IsSticker':'OrderContainsSticker','IsLabel':'OrderContainsLabel','IsPouch':'OrderContainsPouch'}))

first_orders = first_orders.merge(flags, on='OrderNumber', how='left')


first_order_flags = first_orders[['CustomerId', 'ShippingType', 'OrderContainsLabel', 'OrderContainsSticker', 'OrderContainsPouch']].copy()
first_order_flags.rename(columns={
    'OrderContainsLabel': 'FirstOrderContainsLabel',
    'OrderContainsSticker': 'FirstOrderContainsSticker',
    'OrderContainsPouch': 'FirstOrderContainsPouch'
}, inplace=True)

customer_summary = customer_summary.merge(first_order_flags, on='CustomerId', how='left')





if mask_dict:
    for k, v in mask_dict.items():
        customer_summary = customer_summary[customer_summary[k] == v]

prediction_feature = 'OneAndDone'
xgb_features = [
    'AvgOrderItemTotal',
    'AvgItemsPerOrder',
    'HasPromoDiscount',
    'AvgRushFeeAndShipPrice',
    'ShippingType'
    # 'FirstOrderContainsLabel',
]

xgb_data = customer_summary[[prediction_feature] + xgb_features].dropna().copy()

categorical_cols = [
    # 'FirstOrderContainsLabel', 
    'ShippingType',
    'HasPromoDiscount']

label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    xgb_data[col] = le.fit_transform(xgb_data[col].astype(str))
    label_encoders[col] = le

X = xgb_data[xgb_features]
y = xgb_data[prediction_feature]
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

neg, pos = np.bincount(y_train)
scale_pos_weight = neg / pos

xgb = XGBClassifier(
    use_label_encoder=False,
    eval_metric='logloss',
    scale_pos_weight=scale_pos_weight,
    random_state=42
)
xgb.fit(X_train, y_train)

y_probs = xgb.predict_proba(X_test)[:, 1]
threshold = 0.425
y_pred_thresh = (y_probs > threshold).astype(int)

title_suffix = f"\n(Filter: {', '.join([f'{k}={v}' for k, v in mask_dict.items()])})" if mask_dict else ""

conf_matrix = confusion_matrix(y_test, y_pred_thresh)
report_text = classification_report(y_test, y_pred_thresh)

print(classification_report(y_test, y_pred_thresh))


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw={'width_ratios': [1, 1.5]})
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax1,
            xticklabels=['Not Churned', 'Churned'],
            yticklabels=['Not Churned', 'Churned'])
ax1.set_title(f"Confusion Matrix{title_suffix}")
ax1.set_xlabel("Predicted")
ax1.set_ylabel("Actual")
ax2.axis('off')
ax2.text(0, 1, f"Classification Report (threshold = {threshold}):\n\n{report_text}",
         fontsize=11, va='top', family='monospace')
plt.tight_layout()
plt.savefig("outputs/classification_report_and_matrix.png", dpi=300)
plt.show()

importances = xgb.feature_importances_
sorted_idx = np.argsort(importances)[::-1]
plt.figure(figsize=(8, 6))
plt.barh([xgb_features[i] for i in sorted_idx], importances[sorted_idx])
plt.gca().invert_yaxis()
plt.title(f"XGBoost Feature Importances{title_suffix}")
plt.xlabel("Importance")
plt.tight_layout()
plt.show()

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

explainer = shap.Explainer(xgb, X_train)
shap_values = explainer(X_test)
shap.summary_plot(shap_values, X_test, show=False)
plt.tight_layout()
plt.title(f"SHAP Summary Plot{title_suffix}")
plt.savefig("outputs/shap_summary_plot.png", dpi=300)
plt.show()

shap.dependence_plot("AvgOrderItemTotal", shap_values.values, X_test, interaction_index="ShippingType", show=False)
plt.tight_layout(); plt.show()

shap.dependence_plot("ShippingType", shap_values.values, X_test, interaction_index="AvgOrderItemTotal", show=False)
plt.tight_layout(); plt.show()

shap.dependence_plot("AvgOrderItemTotal", shap_values.values, X_test, interaction_index="AvgRushFeeAndShipPrice", show=False)
plt.tight_layout(); plt.show()

shap.dependence_plot("AvgRushFeeAndShipPrice", shap_values.values, X_test, interaction_index="AvgOrderItemTotal", show=False)
plt.tight_layout(); plt.show()


shap_vals = shap_values.values[:, X_test.columns.get_loc('ShippingType')]
shipping = X_test['ShippingType'].values
order_val = X_test['AvgOrderItemTotal'].values
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

plt.figure(figsize=(10, 1.5))
y_jitter = np.random.normal(loc=0, scale=0.05, size=len(shap_vals))
plt.scatter(shap_vals, y_jitter, c=colors, alpha=0.6, s=5)

plt.yticks([])
plt.xlabel("SHAP Value for ShippingType")
plt.title(f"SHAP Value Number Line by Shipping Type and Order Value{title_suffix})")
plt.axvline(0, color='gray', linestyle='--', linewidth=1)
legend_elements = [
    mpatches.Patch(color='purple', label='Normal, < $50'),
    mpatches.Patch(color='green', label='Normal, ≥ $50'),
    mpatches.Patch(color='orange', label='Other, ≥ $75'),
    mpatches.Patch(color='blue', label='Other, < $75'),
]
plt.legend(handles=legend_elements, bbox_to_anchor=(1.02, 1), loc='upper left')
plt.tight_layout()
plt.show()
