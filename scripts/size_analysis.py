import pandas as pd
import os


os.chdir('C:\\Users\\Graphicsland\\Spyder\\retention')

customer_summary = pd.read_feather("C:/Users/Graphicsland/Spyder/retention/outputs/customer_summary.feather")

# customer_summary['LifetimeValue'] = customer_summary['AvgPriceTotal'] * customer_summary['TotalOrders'] 

# Define custom bins — adjust as needed
bins = [0, 100, 500, 1000, 2500, 5000, 10000, 25000, 50000, 250000, float('inf')]
labels = [f"${bins[i]}–{bins[i+1] if bins[i+1] != float('inf') else '∞'}" for i in range(len(bins)-1)]

number_orders_wanted = [1, 2, 3, 4, 5, 6, None]

def generate_revenue_bucket_summary(customer_summary, number_orders_wanted = None):
    # Create spending bucket column
    customer_summary['SpendingBucket'] = pd.cut(
        customer_summary['LifetimeValue'],
        bins=bins,
        labels=labels,
        include_lowest=True
    )
    
    if number_orders_wanted is not None:
        # Filter for customers with the specified number of orders
        subset = customer_summary[customer_summary['TotalOrders'] == number_orders_wanted]
        
        if number_orders_wanted == 1:
            df_name = 'revenue_bucket_summary_1_order'
        else:
            df_name = f'revenue_bucket_summary_{number_orders_wanted}_orders'
    
    else:
        subset = customer_summary
        df_name = 'revenue_bucket_summary_all_orders'

    # Group and aggregate
    revenue_bucket_summary = subset.groupby('SpendingBucket', observed=False).agg(
        CustomerCount=('CustomerId', 'nunique'),
        TotalRevenue=('LifetimeValue', 'sum'),
        ChurnedCount=('LikelyChurned', lambda x: (x == 1).sum())
    ).reset_index()

    # Totals for share calculations
    total_customers = revenue_bucket_summary['CustomerCount'].sum()
    total_revenue = revenue_bucket_summary['TotalRevenue'].sum()

    # Derived percentage columns
    revenue_bucket_summary['Pct_Customers'] = (revenue_bucket_summary['CustomerCount'] / total_customers) * 100
    revenue_bucket_summary['Pct_Revenue'] = (revenue_bucket_summary['TotalRevenue'] / total_revenue) * 100
    revenue_bucket_summary['Pct_ChurnedWithinBucket'] = (
        revenue_bucket_summary['ChurnedCount'] / revenue_bucket_summary['CustomerCount']) * 100

    # Sort bucket order
    revenue_bucket_summary['SpendingBucket'] = pd.Categorical(
        revenue_bucket_summary['SpendingBucket'], categories=labels, ordered=True)
    revenue_bucket_summary = revenue_bucket_summary.sort_values('SpendingBucket')

    # Cumulative percentages
    revenue_bucket_summary['CumulativePct_Customers'] = revenue_bucket_summary['Pct_Customers'].cumsum().round(1)
    revenue_bucket_summary['CumulativePct_Revenue'] = revenue_bucket_summary['Pct_Revenue'].cumsum().round(1)

    # Round final values
    revenue_bucket_summary = revenue_bucket_summary.round({
        'Pct_Customers': 1,
        'Pct_Revenue': 1,
        'Pct_ChurnedWithinBucket': 1
    })

    # Dynamically name the result in the global scope
    globals()[df_name] = revenue_bucket_summary
    return revenue_bucket_summary

for i in number_orders_wanted:
    generate_revenue_bucket_summary(customer_summary, i)




# Define order count bins
order_bins = [1, 2, 3, 4, 5, 6, 10, 20, float('inf')]

# Generate labels with logic for single-integer ranges
order_labels = [
    f"{int(order_bins[i])}" if order_bins[i+1] - order_bins[i] == 1
    else f"{int(order_bins[i])}–{int(order_bins[i+1]) if order_bins[i+1] != float('inf') else '∞'}"
    for i in range(len(order_bins) - 1)
]


customer_summary_high_ltv_subset = customer_summary[customer_summary['LifetimeValue'] >= 1000]

# Create order count bucket
customer_summary_high_ltv_subset['OrderBucket'] = pd.cut(
    customer_summary_high_ltv_subset['TotalOrders'],
    bins=order_bins,
    labels=order_labels,
    include_lowest=True
)

# Group and aggregate
order_bucket_summary = customer_summary_high_ltv_subset.groupby('OrderBucket').agg(
    CustomerCount=('CustomerId', 'nunique'),
    TotalRevenue=('LifetimeValue', 'sum'),
    ChurnedCount=('LikelyChurned', lambda x: (x == 1).sum())
).reset_index()



# Add totals
total_order_customers = order_bucket_summary['CustomerCount'].sum()
total_order_revenue = order_bucket_summary['TotalRevenue'].sum()

# Compute derived columns
order_bucket_summary['Pct_Customers'] = (order_bucket_summary['CustomerCount'] / total_order_customers) * 100
order_bucket_summary['Pct_Revenue'] = (order_bucket_summary['TotalRevenue'] / total_order_revenue) * 100
order_bucket_summary['Pct_ChurnedWithinBucket'] = (
    order_bucket_summary['ChurnedCount'] / order_bucket_summary['CustomerCount']) * 100

# Sort buckets in defined order
order_bucket_summary['OrderBucket'] = pd.Categorical(order_bucket_summary['OrderBucket'], categories=order_labels, ordered=True)
order_bucket_summary = order_bucket_summary.sort_values('OrderBucket')

# Add cumulative percent columns
order_bucket_summary['CumulativePct_Customers'] = order_bucket_summary['Pct_Customers'].cumsum().round(1)
order_bucket_summary['CumulativePct_Revenue'] = order_bucket_summary['Pct_Revenue'].cumsum().round(1)

# Round percentages
order_bucket_summary = order_bucket_summary.round({
    'Pct_Customers': 1,
    'Pct_Revenue': 1,
    'Pct_ChurnedWithinBucket': 1
})


order_bucket_summary.to_csv("outputs/order_bucket_summary_high_ltv.csv", index=False)
# order_bucket_summary.to_csv("outputs/order_bucket_summary_high_ltv_500_or_more.csv", index=False) # Semi-manual
# order_bucket_summary.to_csv("outputs/order_bucket_summary_all.csv", index=False) # Semi-manual
revenue_bucket_summary_1_order.to_csv("outputs/revenue_bucket_summary_1_order.csv", index=False)
revenue_bucket_summary_2_orders.to_csv("outputs/revenue_bucket_summary_2_orders.csv", index=False)
revenue_bucket_summary_3_orders.to_csv("outputs/revenue_bucket_summary_3_orders.csv", index=False)
revenue_bucket_summary_all_orders.to_csv("outputs/revenue_bucket_summary_all_orders.csv", index=False)


