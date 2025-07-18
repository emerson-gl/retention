import pandas as pd
import numpy as np
import os

os.chdir("C:/Users/Graphicsland/Spyder/retention")

customer_summary = pd.read_feather("outputs/customer_summary.feather")
order_df_retention = pd.read_feather("outputs/order_df_retention.feather")

# -----------------------------------
# Tag ProductType (duplicating rows for mixed customers)
# -----------------------------------
customer_summary['ProductType_Sticker'] = customer_summary['PctOrdersWithSticker'].apply(lambda x: 'Sticker' if x > 0 else None)
customer_summary['ProductType_Label'] = customer_summary['PctOrdersWithLabel'].apply(lambda x: 'Label' if x > 0 else None)

product_type_df = pd.melt(
    customer_summary,
    id_vars=[col for col in customer_summary.columns if not col.startswith("ProductType_")],
    value_vars=['ProductType_Sticker', 'ProductType_Label'],
    value_name='ProductType'
).dropna(subset=['ProductType']).drop(columns='variable')

product_type_df['FirstOrderDate'] = pd.to_datetime(product_type_df['FirstOrderDate'])

# -----------------------------------
# Set up order buckets
# -----------------------------------
order_bins = [1, 2, 3, 4, 5, 6, 10, 20, float("inf")]
order_labels = [
    f"{int(order_bins[i])}" if order_bins[i+1] - order_bins[i] == 1
    else f"{int(order_bins[i])}–{int(order_bins[i+1]) if order_bins[i+1] != float('inf') else '∞'}"
    for i in range(len(order_bins) - 1)
]
product_type_df['OrderBucket'] = pd.cut(product_type_df['TotalOrders'], bins=order_bins, labels=order_labels, include_lowest=True)

# -----------------------------------
# Filter 2025 shipped orders for monthly activity
# -----------------------------------
order_df_retention['ShippedDateTime'] = pd.to_datetime(order_df_retention['ShippedDateTime'])


# -----------------------------------
# Function for Order Bucket Summary
def generate_order_bucket_summary(df, suffix):
    # Compute monthly ordering customers for 2025 Jan–Jun
    eligible_customers = df[['CustomerId', 'ProductType', 'OrderBucket']].drop_duplicates()
    merged_orders = order_df_retention.merge(eligible_customers, on='CustomerId', how='inner')
    
    orders_2025 = merged_orders[
        (merged_orders['IsDeleted'] != 1) &
        (merged_orders['ShippedDateTime'].notna()) &
        (merged_orders['ShippedDateTime'].dt.year == 2025) &
        (merged_orders['ShippedDateTime'].dt.month <= 6)
    ].copy()
    
    orders_2025['Month'] = orders_2025['ShippedDateTime'].dt.month

    monthly_presence = (
        orders_2025
        .groupby(['ProductType', 'OrderBucket', 'Month'], observed=False)['CustomerId']
        .nunique()
        .reset_index(name='CustsThisMonth')
    )

    avg_monthly = (
        monthly_presence
        .groupby(['ProductType', 'OrderBucket'], observed=False)['CustsThisMonth']
        .mean()
        .round(1)
        .reset_index(name='AvgMonthlyOrderingCusts_2025')
    )

    # Main summary
    summary = (
        df
        .groupby(['ProductType', 'OrderBucket'], observed=False)
        .agg(
            CustomerCount=('CustomerId', 'nunique'),
            TotalRevenue=('LifetimeValue', 'sum'),
            ChurnedCount=('LikelyChurned', lambda x: (x == 1).sum())
        )
        .reset_index()
    )

    summary = summary.merge(avg_monthly, on=['ProductType', 'OrderBucket'], how='left')
    summary['Pct_Customers'] = summary.groupby('ProductType')['CustomerCount'].transform(lambda x: x / x.sum())
    summary['Pct_Revenue'] = summary.groupby('ProductType')['TotalRevenue'].transform(lambda x: x / x.sum())
    summary['Pct_ChurnedWithinBucket'] = summary['ChurnedCount'] / summary['CustomerCount']
    summary['OrderBucket'] = pd.Categorical(summary['OrderBucket'], categories=order_labels, ordered=True)
    summary = summary.sort_values(['ProductType', 'OrderBucket'])
    summary['CumulativePct_Customers'] = summary.groupby('ProductType')['Pct_Customers'].cumsum().round(3)
    summary['CumulativePct_Revenue'] = summary.groupby('ProductType')['Pct_Revenue'].cumsum().round(3)

    summary = summary.round({
        'Pct_Customers': 3,
        'Pct_Revenue': 3,
        'Pct_ChurnedWithinBucket': 3
    })

    for product in summary['ProductType'].unique():
        summary[summary['ProductType'] == product].to_csv(f"outputs/order_bucket_summary_{product.lower()}_{suffix}.csv", index=False)

# -----------------------------------
# Generate for all / LTV ≥ 500 / LTV ≥ 1000
# -----------------------------------
generate_order_bucket_summary(product_type_df, 'all')
generate_order_bucket_summary(product_type_df[product_type_df['LifetimeValue'] >= 500], '500plus')
generate_order_bucket_summary(product_type_df[product_type_df['LifetimeValue'] >= 1000], '1000plus')


# -----------------------------------
# Revenue Bucket Summary Function
# -----------------------------------


revenue_bins = [0, 100, 500, 1000, 2500, 5000, 10000, 25000, 50000, 250000, float('inf')]
revenue_labels = [f"${revenue_bins[i]}–{revenue_bins[i+1] if revenue_bins[i+1] != float('inf') else '∞'}" for i in range(len(revenue_bins)-1)]

def generate_revenue_bucket_summary(df, order_df_retention, suffix):
    df = df.copy()
    df['FirstOrderDate'] = pd.to_datetime(df['FirstOrderDate'])
    df['SpendingBucket'] = pd.cut(df['LifetimeValue'], bins=revenue_bins, labels=revenue_labels, include_lowest=True)

    summaries = {}

    for product in ['Sticker', 'Label']:
        subset = df[df['ProductType'] == product].copy()

        # Deduplicate eligible customers by CustomerId and SpendingBucket
        eligible_customers = subset[['CustomerId', 'ProductType', 'SpendingBucket']].drop_duplicates()

        # Filter valid shipped orders in Jan–Jun 2025
        orders_2025 = (
            order_df_retention.merge(eligible_customers, on='CustomerId', how='inner')
            .query("IsDeleted != 1 and ShippedDateTime.notna()")
            .assign(ShippedDateTime=lambda x: pd.to_datetime(x['ShippedDateTime']))
        )
        orders_2025 = orders_2025[
            (orders_2025['ShippedDateTime'].dt.year == 2025) &
            (orders_2025['ShippedDateTime'].dt.month <= 6)
        ].copy()

        orders_2025['Month'] = orders_2025['ShippedDateTime'].dt.month

        # Monthly ordering presence per bucket
        monthly_presence = (
            orders_2025
            .groupby(['SpendingBucket', 'Month'], observed=False)['CustomerId']
            .nunique()
            .reset_index(name='CustsThisMonth')
        )

        avg_monthly = (
            monthly_presence
            .groupby(['SpendingBucket'], observed=False)['CustsThisMonth']
            .mean()
            .round(1)
            .reset_index(name='AvgMonthlyOrderingCusts_2025')
        )

        # Summary aggregation
        summary = subset.groupby('SpendingBucket', observed=False).agg(
            CustomerCount=('CustomerId', 'nunique'),
            TotalRevenue=('LifetimeValue', 'sum'),
            ChurnedCount=('LikelyChurned', lambda x: (x == 1).sum())
        ).reset_index()

        summary = summary.merge(avg_monthly, on='SpendingBucket', how='left')

        summary['Pct_Customers'] = summary['CustomerCount'] / summary['CustomerCount'].sum()
        summary['Pct_Revenue'] = summary['TotalRevenue'] / summary['TotalRevenue'].sum()
        summary['Pct_ChurnedWithinBucket'] = summary['ChurnedCount'] / summary['CustomerCount']

        summary['SpendingBucket'] = pd.Categorical(summary['SpendingBucket'], categories=revenue_labels, ordered=True)
        summary = summary.sort_values('SpendingBucket')

        summary['CumulativePct_Customers'] = summary['Pct_Customers'].cumsum().round(3)
        summary['CumulativePct_Revenue'] = summary['Pct_Revenue'].cumsum().round(3)

        summary = summary.round({
            'Pct_Customers': 3,
            'Pct_Revenue': 3,
            'Pct_ChurnedWithinBucket': 3
        })

        summary.to_csv(f"outputs/revenue_bucket_summary_{product.lower()}_{suffix}.csv", index=False)
        summaries[product] = summary

    return summaries

generate_revenue_bucket_summary(product_type_df, order_df_retention, 'all')