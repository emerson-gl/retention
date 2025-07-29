import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt


# ─────────────── SETUP ───────────────
os.chdir('C:/Users/Graphicsland/Spyder/retention')
cutoff_date = pd.Timestamp('2023-01-01')
mask_dict = {'PctOrdersWithSticker': 1}
# mask_dict = {'PctOrderContainsLabel': 1}
minimum_n = 500


# ───────────────────────── 1  LOAD + CLEAN ─────────────────────────
order_df = pd.read_feather('../sharedData/raw_order_df.feather')
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
    .first()
)

first_orders['ShippingType'] = np.select(
    [
        (first_orders['OrderItemPriceTotal'] < 75) & (first_orders['ShippingTotal'] == 0),
        (first_orders['OrderItemPriceTotal'] >= 75) & (first_orders['ShippingTotal'] == 0),
        (first_orders['OrderItemPriceTotal'] < 75) & (first_orders['ShippingTotal'] == 4),
        (first_orders['ShippingTotal'] > 4)
    ],
    [
        'NormalShipping',
        'TwoDayShipping',
        'TwoDayShipping',
        'ExpeditedShipping'
    ],
    default='UncategorizedShipping'
)

# Get list or set of relevant OrderNumbers
order_numbers = set(first_orders['OrderNumber'].astype(str))

# Now select relevant columns

# ─────────────── LOAD ───────────────
customer_summary = pd.read_feather("outputs/customer_summary_first_order.feather")

if mask_dict:
    for k, v in mask_dict.items():
        customer_summary = customer_summary[customer_summary[k] == v]

bq_df = pd.read_feather('../sharedData/bigquery_events.feather')

bq_df_subset = bq_df[bq_df['param_transaction_id'].astype(str).isin(order_numbers)]
bq_df_subset = bq_df_subset[['param_transaction_id', 'traffic_source_medium', 'traffic_source_source', 'traffic_source_name', 'param_medium', 'param_source', 'device_category', 'device_operating_system', 'geo_metro', ]]

def most_common(series):
    return series.dropna().mode().iloc[0] if not series.dropna().empty else None

bq_df_collapsed = bq_df.groupby('param_transaction_id').agg({
    'traffic_source_medium': most_common,
    'traffic_source_source': most_common,
    'traffic_source_name': most_common,
    'param_medium': most_common,
    'param_source': most_common,
    'device_category': most_common,
    'device_operating_system': most_common,
    'geo_metro': most_common,
}).reset_index()


first_orders['OrderNumber_str'] = first_orders['OrderNumber'].astype(str)
bq_df_collapsed['param_transaction_id_str'] = bq_df_collapsed['param_transaction_id'].astype(str)

first_orders_bq = first_orders.merge(
    bq_df_collapsed, 
    left_on='OrderNumber_str', 
    right_on='param_transaction_id_str', 
    how='left'
)

first_orders_bq = first_orders_bq.merge(
    customer_summary[['CustomerId', 'OneAndDone']],
    on='CustomerId',
    how='right')


first_orders_bq['OneAndDone'] = first_orders_bq['OneAndDone'].astype(int)


cols_to_check = [
    'traffic_source_medium',
    'traffic_source_source',
    'traffic_source_name',
    'param_medium',
    'param_source',
    'device_category',
    'device_operating_system',
    'geo_metro'
]

for col in cols_to_check:
    if col not in first_orders_bq.columns:
        print(f"Skipping missing column: {col}")
        continue

    # Get value counts and filter to those with at least
    
    value_counts = first_orders_bq[col].value_counts()
    eligible_values = value_counts[value_counts >= minimum_n].index
    filtered = first_orders_bq[first_orders_bq[col].isin(eligible_values)]

    print(f"\n--- {col} ---")
    print(f"Eligible values (n≥{minimum_n}):\n{eligible_values}")
    print(f"Filtered row count: {len(filtered)}")
    print(f"OneAndDone counts:\n{filtered['OneAndDone'].value_counts(dropna=False)}")

    if filtered['OneAndDone'].nunique() == 0:
        print(f"No OneAndDone data for {col}, skipping plot.")
        continue

    # Crosstab and label with n
    crosstab = pd.crosstab(filtered[col], filtered['OneAndDone'], normalize='index')
    counts = filtered[col].value_counts()
    crosstab['label'] = crosstab.index.astype(str) + '\n(n=' + crosstab.index.map(counts).astype(str) + ')'

    # Sort descending by OneAndDone=1 rate (or just True if bool)
    sort_col = 1 if 1 in crosstab.columns else True
    crosstab_sorted = crosstab.sort_values(by=sort_col, ascending=False)

    # Plot
    plot_data = crosstab_sorted.drop(columns='label')
    plot_data.index = crosstab_sorted['label']

    ax = plot_data.plot(kind='bar', stacked=True, figsize=(14, 7), colormap='viridis')
    plt.title(f'OneAndDone by {col} (n≥{minimum_n})')
    plt.ylabel('Proportion')
    plt.xlabel(col)
    plt.legend(title='OneAndDone', loc='upper right')
    plt.tight_layout()
    plt.show()
