import pandas as pd
import os

# Set working directory
os.chdir('C:\\Users\\Graphicsland\\Spyder\\retention')

# Load data
order_df = pd.read_feather("C:/Users/Graphicsland/Spyder/sharedData/raw_order_df.feather")
order_item_df = pd.read_feather("C:/Users/Graphicsland/Spyder/sharedData/raw_order_item_df.feather")
ppo = pd.read_feather("C:/Users/Graphicsland/Spyder/sharedData/raw_product_product_option_df.feather")
customer_summary = pd.read_feather("C:/Users/Graphicsland/Spyder/retention/outputs/customer_summary_first_2_order.feather")


# Filter to relevant affiliate
order_df = order_df[order_df['AffiliateId'] == 2]

# Identify sample product option IDs
sample_ids = ppo[['Id', 'SKU']][ppo['SKU'].str.contains('ample', na=False)]
sample_id_set = set(sample_ids['Id'])

# Mark sample items in order_item_df
order_item_df['IsSample'] = order_item_df['ProductProductOptionId'].isin(sample_id_set)

# Join order_item_df with order_df to get CustomerId and OrderDate
order_items_with_customer = order_item_df.merge(
    order_df[['OrderNumber', 'CustomerId', 'OrderDate']],
    on='OrderNumber',
    how='left'
)

# Find each customer's first order
first_order_dates = order_items_with_customer.groupby('CustomerId')['OrderDate'].min().reset_index()
first_order_items = order_items_with_customer.merge(first_order_dates, on=['CustomerId', 'OrderDate'])

# Flag: all items in first order are samples
first_order_flags = first_order_items.groupby('CustomerId')['IsSample'].all().reset_index()
first_order_flags.rename(columns={'IsSample': 'FirstOrderSampleOnly'}, inplace=True)

# Flag: has ever ordered a sample
has_sample_flags = order_items_with_customer.groupby('CustomerId')['IsSample'].any().reset_index()
has_sample_flags.rename(columns={'IsSample': 'HasOrderedSample'}, inplace=True)

# Merge into customer_summary
customer_summary = customer_summary.merge(first_order_flags, on='CustomerId', how='left')
customer_summary = customer_summary.merge(has_sample_flags, on='CustomerId', how='left')

# Fill NAs for customers with no orders
customer_summary['FirstOrderSampleOnly'] = customer_summary['FirstOrderSampleOnly'].fillna(False)
customer_summary['HasOrderedSample'] = customer_summary['HasOrderedSample'].fillna(False)


# Save updated summary
customer_summary.to_feather("outputs/customer_summary_first_2_order.feather")
