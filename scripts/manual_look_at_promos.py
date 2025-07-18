import pandas as pd
import numpy as np
import os

# Set working directory and load data
os.chdir("C:/Users/Graphicsland/Spyder/retention")
product_type_df = pd.read_feather("outputs/product_type_df.feather")  # Or generate from customer_summary
order_df_retention = pd.read_feather("outputs/order_df_retention.feather")

# Filter to high LTV label customers in 21–∞ bucket
high_ltv_label = product_type_df[
    (product_type_df['ProductType'] == 'Label') &
    (product_type_df['LifetimeValue'] >= 1000)
].copy()

# Define order bucket again if needed
order_bins = [1, 2, 3, 4, 5, 6, 10, 20, float('inf')]
order_labels = [
    f"{int(order_bins[i])}" if order_bins[i+1] - order_bins[i] == 1
    else f"{int(order_bins[i])}–{int(order_bins[i+1]) if order_bins[i+1] != float('inf') else '∞'}"
    for i in range(len(order_bins) - 1)
]
high_ltv_label['OrderBucket'] = pd.cut(high_ltv_label['TotalOrders'], bins=order_bins, labels=order_labels, include_lowest=True)

# Filter to 21–∞
# high_ltv_label_21plus = high_ltv_label[high_ltv_label['OrderBucket'] == '1']
high_ltv_label_21plus = high_ltv_label[high_ltv_label['OrderBucket'] == '20–∞']

# Get CustomerIds
cust_ids = high_ltv_label_21plus['CustomerId'].unique()

# Filter orders
orders = order_df_retention[
    (order_df_retention['CustomerId'].isin(cust_ids)) &
    (order_df_retention['IsDeleted'] != 1) &
    (order_df_retention['ShippedDateTime'].notna())
]

# Calculate % of orders with promo codes
total_orders = len(orders)
orders_with_promo = (orders['PromoCodeDiscount'].abs() > 0).sum()
pct_with_promo = round(orders_with_promo / total_orders * 100, 1)

print(f"% of orders with a promo code for 21+ order high-LTV label customers: {pct_with_promo}%")
