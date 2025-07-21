import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree

# ─────────────── SETUP ───────────────
os.chdir('C:/Users/Graphicsland/Spyder/retention')
cutoff_date = pd.Timestamp('2023-01-01')
mask_dict = {'FirstOrderContainsLabel': True}

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


def label_leaf(row):
    items = row['AvgItemsPerOrder']
    total = row['AvgOrderItemTotal']
    promo = row['PctUsedPromo']
    churn = row['PctOneAndDone']
    
    # One-and-done zone: very low item count and low order total
    if items < 1.5 and total < 100:
        return 'One-and-Done Zone'

    # Bargain Browsers: low order value, low item count, uses promo
    if items < 3 and total < 250 and promo > 0.5:
        return 'Bargain Browser'
    
    # High Rollers: large orders, many items, low promo use
    if items > 6 and total > 400 and promo < 0.3:
        return 'High Roller'

    # Bulk Buyers: high item count, moderate-high total
    if items > 5 and total > 200:
        return 'Bulk Buyer'

    # Steady Shippers: moderate behavior
    if 2 <= items <= 6 and 100 <= total <= 400 and promo < 0.5 and churn < 0.6:
        return 'Steady Shipper'
    
    # Otherwise: fallback
    return 'Uncategorized'



# Apply labeling function
summary['LeafLabel'] = summary.apply(label_leaf, axis=1)


def refine_uncategorized(row):
    if row['LeafLabel'] != 'Uncategorized':
        return row['LeafLabel']
    
    if row['PctUsedPromo'] >= 0.9:
        return 'Promo Seeker'
    
    if row['PctUsedPromo'] <= 0.1:
        return 'Non-Promo Explorer'
    
    if row['AvgOrderItemTotal'] < 150:
        return 'Low-Value Mid-Spender'
    
    return 'Uncategorized Refined'

summary['RefinedLeafLabel'] = summary.apply(refine_uncategorized, axis=1)




grouped_refined = summary.groupby('RefinedLeafLabel').agg({
    'LeafSize': 'sum',
    'LeafValue': lambda x: np.sum([float(str(v).replace('$','').replace(',','')) for v in x]),
    'AvgItemsPerOrder': 'mean',
    'AvgOrderItemTotal': 'mean',
    'AvgRushFeeAndShipPrice': 'mean',
    'PctUsedPromo': 'mean',
    'PctOneAndDone': 'mean'
}).rename(columns={
    'LeafSize': 'TotalCustomers',
    'LeafValue': 'TotalValue',
    'AvgItemsPerOrder': 'AvgItemsPerOrder',
    'AvgOrderItemTotal': 'AvgOrderItemTotal',
    'AvgRushFeeAndShipPrice': 'AvgRushFeeAndShipPrice',
    'PctUsedPromo': 'AvgPctUsedPromo',
    'PctOneAndDone': 'AvgPctOneAndDone'
}).reset_index()
