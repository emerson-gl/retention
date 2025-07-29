import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree

# ─────────────── SETUP ───────────────
os.chdir('C:/Users/Graphicsland/Spyder/retention')
cutoff_date = pd.Timestamp('2023-01-01')
mask_dict = {'PctOrdersWithSticker': 1}
mask_dict = {'PctOrdersWithLabel': 1}

# ─────────────── LOAD ───────────────
customer_summary = pd.read_feather("outputs/customer_summary_first_order.feather")

order_df = pd.read_feather('../sharedData/raw_order_df.feather')
order_df = (
    order_df[(order_df['IsDeleted'] != 1) & (order_df['AffiliateId'] == 2)]
    .dropna(subset=['ShippedDateTime'])
    .copy()
)

item_df  = pd.read_feather('../sharedData/raw_order_item_df.feather')
item_df = (
    item_df[(item_df['IsDeleted'] != 1)]
    .copy()
)

ppo      = pd.read_feather('../sharedData/raw_product_product_option_df.feather')

# ─────────────── CLEAN ───────────────
ppo['IsSticker'] = ppo['Name'].str.contains('icker', case=False, na=False) & ~ppo['Name'].str.contains('ample', case=False, na=False)
ppo['IsLabel']   = ppo['Name'].str.contains('abel', case=False, na=False) & ~ppo['Name'].str.contains('ample', case=False, na=False)
ppo['IsPouch']   = ppo['Name'].str.contains('ouch', case=False, na=False) & ~ppo['Name'].str.contains('ample', case=False, na=False)

item_df = item_df.merge(ppo[['Id', 'IsSticker', 'IsLabel', 'IsPouch']],
                        left_on='ProductProductOptionId', right_on='Id', how='left')



first_orders = (
    order_df
    .sort_values('PlacedDateTime')
    .groupby('CustomerId', as_index=False)
    .first()
)

first_orders['ShippingType'] = np.select(
    [(first_orders['OrderItemPriceTotal'] < 75) & (first_orders['ShippingTotal'] == 0)],
    ['NormalShipping'],
    default='OtherShipping'
)

flags = (
    item_df.groupby('OrderNumber')[['IsSticker', 'IsLabel', 'IsPouch']]
    .any()
    .rename(columns={
        'IsSticker': 'OrderContainsSticker',
        'IsLabel': 'OrderContainsLabel',
        'IsPouch': 'OrderContainsPouch'
    })
)

first_orders = first_orders.merge(flags, on='OrderNumber', how='left')

first_order_flags = first_orders[['CustomerId', 'ShippingType', 'OrderContainsLabel', 'OrderContainsSticker', 'OrderContainsPouch']].copy()
first_order_flags.rename(columns={
    'OrderContainsLabel': 'FirstOrderContainsLabel',
    'OrderContainsSticker': 'FirstOrderContainsSticker',
    'OrderContainsPouch': 'FirstOrderContainsPouch'
}, inplace=True)

customer_summary = customer_summary.merge(first_order_flags, on='CustomerId', how='left')
customer_summary = customer_summary[customer_summary['FirstOrderDate'] >= cutoff_date]

for k, v in mask_dict.items():
    customer_summary = customer_summary[customer_summary[k] == v]

# ─────────────── TREE FEATURES ───────────────
tree_features = [
    'AvgOrderItemTotal',
    'AvgItemsPerOrder',
    'HasPromoDiscount',
    'ShippingType',
    'AvgRushFeeAndShipPrice',
]
target_var = 'OneAndDone'

customer_summary['HasPromoDiscount'] = customer_summary['AvgPromoDiscount'] < 0
tree_data = customer_summary[['CustomerId', target_var] + tree_features].dropna().copy()

# Prepare features
X = tree_data[tree_features].copy()
X = pd.get_dummies(X, columns=['ShippingType'], drop_first=True)
y = tree_data[target_var]

# ─────────────── FIT DECISION TREE ───────────────
tree_clf = DecisionTreeClassifier(max_depth=4, min_samples_leaf=20, random_state=42)
tree_clf.fit(X, y)

# Assign each sample to a leaf node
tree_data['LeafID'] = tree_clf.apply(X)

# ─────────────── SUMMARIZE LEAVES ───────────────
summary = tree_data.groupby('LeafID').agg(
    LeafSize=('CustomerId', 'count'),
    AvgItemsPerOrder=('AvgItemsPerOrder', 'mean'),
    AvgOrderItemTotal=('AvgOrderItemTotal', 'mean'),
    AvgRushFeeAndShipPrice=('AvgRushFeeAndShipPrice', 'mean'),
    PctUsedPromo=('HasPromoDiscount', 'mean'),
    PctOneAndDone=('OneAndDone', 'mean')
).round(2)

summary['LeafValue'] = summary['LeafSize'] * summary['AvgOrderItemTotal']
summary['LeafValue'] = summary['LeafValue'].apply(lambda x: f"${x:,.2f}")

# Reorder for readability
cols = list(summary.columns)
cols.insert(cols.index('AvgOrderItemTotal') + 1, cols.pop(cols.index('LeafValue')))
summary = summary[cols]

# ─────────────── PLOT DECISION TREE ───────────────
plt.figure(figsize=(20, 8))
plot_tree(tree_clf, feature_names=X.columns, class_names=['Retained', 'OneAndDone'], filled=True, rounded=True)
plt.title("Decision Tree Predicting OneAndDone")
plt.show()

def name_leaf_sticker(row):
    items = row['AvgItemsPerOrder']
    total = row['AvgOrderItemTotal']
    promo = row['HasPromoDiscount']
    
    # 📦 Large recurring orders
    if (items > 3 and total > 175) or total > 250 or items > 6:
        return 'Bulk Buyer'

    # 🚫 Clear churn risk
    if items < 3 and total < 75:
        return 'One-and-Done Zone'

    # 🧠 Bargain-focused
    if items < 3 and total < 250 and promo == 1:
        return 'Bargain Shopper'

    # 🛍️ Value-conscious but retained
    if 2 <= items <= 6 and 100 <= total <= 450:
        return 'Value Seeker'

    # 📬 Normal promo-free shoppers
    if 2 <= items <= 6:
        return 'Normal Shopper'


    return 'Uncategorized'


def name_leaf_label(row):
    items = row['AvgItemsPerOrder']
    total = row['AvgOrderItemTotal']
    promo = row['HasPromoDiscount']

    # High volume, high total — relaxed Bulk Buyer
    if total > 300:
        return 'Bulk Buyer'

    # Very low item count, very low promo — One-and-Done
    if items < 3 and total < 100:
        return 'One-and-Done Zone'

    # Moderate item count, moderate total, low promo — Steady Shipper
    if  items > 2 and total > 100:
        return 'Steady Shipper'

    # Mid to low quantity shoppers with moderate promo — Bargain Browser
    if  items > 2 and total < 100:
        return 'Low value high items'

    # Zero promo usage — Non-Promo Explorer
    if promo == 0:
        return 'Non-Promo Explorer'

    # Everything else — fallback
    return 'Selective Spender'


def apply_segment_naming(row, mask_dict):
    if mask_dict.get('FirstOrderContainsSticker', False):
        return name_leaf_sticker(row)
    elif mask_dict.get('FirstOrderContainsLabel', False):
        return name_leaf_label(row)
    else:
        return 'Uncategorized'

customer_summary['SegmentName'] = customer_summary.apply(lambda row: apply_segment_naming(row, mask_dict), axis=1)


def grouped_summary_from_customer_summary(df):
    grouped = (
        df
        .groupby('SegmentName')
        .agg(
            TotalCustomers=('CustomerId', 'count'),
            TotalValue=('AvgOrderItemTotal', 'sum'),
            AvgItemsPerOrder=('AvgItemsPerOrder', 'mean'),
            AvgOrderItemTotal=('AvgOrderItemTotal', 'mean'),
            MedianOrderItemTotal=('AvgOrderItemTotal', 'median'),
            AvgRushFeeAndShipPrice=('AvgRushFeeAndShipPrice', 'mean'),
            AvgPctUsedPromo=('HasPromoDiscount', 'mean'),
            AvgPctOneAndDone=('OneAndDone', 'mean'),
        )
        .reset_index()
    )

    # Formatting
    
    grouped['TotalValue'] = grouped['TotalValue'].apply(lambda x: f"${x:,.0f}")
    grouped['AvgPctUsedPromo'] = grouped['AvgPctUsedPromo'].apply(lambda x: f"{x * 100:.1f}%")
    grouped['AvgPctOneAndDone'] = grouped['AvgPctOneAndDone'].apply(lambda x: f"{x * 100:.1f}%")

    return grouped

grouped_summary = grouped_summary_from_customer_summary(customer_summary)

