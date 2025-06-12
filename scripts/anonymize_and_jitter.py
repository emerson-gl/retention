import pandas as pd
import numpy as np
import hashlib
import os
from datetime import timedelta


n_customers = 10000

os.chdir('C:\\Users\\Graphicsland\\Spyder\\retention')

# Load your raw data
order_df = pd.read_feather("C:/Users/Graphicsland/Spyder/sharedData/raw_order_df.feather")
order_item_df = pd.read_feather("C:/Users/Graphicsland/Spyder/sharedData/raw_order_item_df.feather")

# Filter orders for MakeStickers + shipped before March 2025
order_df = order_df[
    (order_df['AffiliateId'] == 2) &
    (order_df['IsDeleted'] != True) &
    (pd.to_datetime(order_df['ShippedDateTime'], errors='coerce') < pd.Timestamp("2025-03-01"))
].copy()

# Filter order items
order_item_df = order_item_df[order_item_df['IsDeleted'] != True].copy()

# Count items per order
item_counts = order_item_df.groupby('OrderNumber').size().reset_index(name='TotalOrderItems')

# Select fields
cols_to_keep = [
    'OrderNumber', 'CustomerId', 'PlacedDateTime', 'PaymentMethod',
    'OrderItemPriceTotal', 'RushFee', 'Tax', 'ShipPrice', 'PromoCodeDiscount', 'NoPromotionalDesignUse'
]
merged_df = order_df[cols_to_keep].merge(item_counts, on='OrderNumber', how='left')

# Salted hash function
salt = "proud_2_p4rt4ke_of_your_pec4n_314159265358979323846264338327950288"
def salted_hash(value, salt):
    if pd.isna(value):
        return None
    return hashlib.sha256(f"{salt}_{value}".encode()).hexdigest()

# Apply salted hashes
merged_df['OrderNumberHash'] = merged_df['OrderNumber'].apply(lambda x: salted_hash(x, salt))
merged_df['CustomerIdHash'] = merged_df['CustomerId'].apply(lambda x: salted_hash(x, salt))

# Jitter PlacedDateTime by up to 14 days back
jitter_days = np.random.randint(0, 15, size=len(merged_df))
merged_df['PlacedDateTime'] = pd.to_datetime(merged_df['PlacedDateTime'], errors='coerce') - pd.to_timedelta(jitter_days, unit='d')

# Jitter numeric amounts (up/down ±10%)
def jitter_amount(val):
    if pd.isna(val) or val == 0:
        return val
    return val * (1 + np.random.uniform(-0.1, 0.1))

for col in ['OrderItemPriceTotal', 'RushFee', 'Tax', 'ShipPrice', 'PromoCodeDiscount']:
    merged_df[col] = merged_df[col].apply(jitter_amount)

# Obfuscate some PaymentMethod and NoPromotionalDesignUse values
def obfuscate_column(df, col, pct=0.15):
    unique_vals = df[col].dropna().unique()
    mask = np.random.rand(len(df)) < pct
    df.loc[mask, col] = np.random.choice(unique_vals, size=mask.sum())
    return df

merged_df = obfuscate_column(merged_df, 'PaymentMethod')
merged_df = obfuscate_column(merged_df, 'NoPromotionalDesignUse')

# Save hashed reference for internal use
hash_reference = merged_df[['OrderNumber', 'OrderNumberHash', 'CustomerId', 'CustomerIdHash']]
hash_reference.to_feather("hash_reference.feather")

# Drop real IDs before external sharing
removed_ids_df = merged_df.drop(columns=['OrderNumber', 'CustomerId'])

# Sample a subset of customers
sampled_customer_ids = removed_ids_df['CustomerId'].dropna().drop_duplicates().sample(n=n_customers, random_state=42)
export_df = removed_ids_df[removed_ids_df['CustomerId'].isin(sampled_customer_ids)].drop(columns=['OrderNumber', 'CustomerId'])


# Count number of orders per customer
order_counts = export_df['CustomerIdHash'].value_counts()

# Count how many customers placed 1, 2, 3, ..., N orders
order_frequency_distribution = order_counts.value_counts().sort_index()

# Display result
print(order_frequency_distribution)

# top_customer = export_df['CustomerIdHash'].value_counts().idxmax()
# export_df = export_df[export_df['CustomerIdHash'] != top_customer]


export_df.to_csv("outputs/pecan_export.csv", index=False)
