import pandas as pd
import numpy as np
import os
from datetime import timedelta
from joblib import Parallel, delayed
import gc


# Decide if you want first or last orders
first_or_last = 'first'
# first_or_last = 'last'
first_or_last = None

# Decide how many orders
number_of_orders = 1 #if using 1, only run through line ~232 and save agg_df as customer_summary_[first/last]_order. Janky, I know.



# -------------------------------------------------------------------------
# 1) LOAD AND CLEAN
# -------------------------------------------------------------------------
os.chdir('C:\\Users\\Graphicsland\\Spyder\\retention')

# Load data
order_df = pd.read_feather("C:/Users/Graphicsland/Spyder/sharedData/raw_order_df.feather")
order_item_df = pd.read_feather("C:/Users/Graphicsland/Spyder/sharedData/raw_order_item_df.feather")
ppo_df = pd.read_feather("C:/Users/Graphicsland/Spyder/sharedData/raw_product_product_option_df.feather")

# Create product flags
ppo_df['IsSticker'] = (
    ppo_df['Name'].str.contains('icker', case=False, na=False) &
    ~ppo_df['Name'].str.contains('ample', case=False, na=False)
)

ppo_df['IsLabel'] = (
    ppo_df['Name'].str.contains('abel', case=False, na=False) &
    ~ppo_df['Name'].str.contains('ample', case=False, na=False)
)

order_item_df = order_item_df.merge(
    ppo_df[['Id', 'IsSticker', 'IsLabel']],
    left_on='ProductProductOptionId',
    right_on='Id',
    how='left'
)

# Count of non-deleted items per order
item_count = order_item_df.groupby('OrderNumber').size().reset_index(name='ItemCount')
order_df = order_df.merge(item_count, on='OrderNumber', how='left')




# Clean
order_df = order_df[
    (order_df['IsDeleted'] != True) &
    (order_df['AffiliateId'] == 2) &
    (order_df['ShippedDateTime'].notna())
].copy()
order_item_df = order_item_df[order_item_df['IsDeleted'] != True].copy()
# feedback_df feedback_df[feedback_df['IsDeleted'] != True].copy() # Not sure

# Dates and times
order_df['OrderDate'] = pd.to_datetime(order_df['OrderDate'], errors='coerce')
order_df['PlacedDateTime'] = pd.to_datetime(order_df['PlacedDateTime'], errors='coerce')
order_df['ShippedDateTime'] = pd.to_datetime(order_df['ShippedDateTime'], errors='coerce')
order_df['ShipDelayHours'] = (order_df['ShippedDateTime'] - order_df['PlacedDateTime']).dt.total_seconds() / 3600
order_df['HourOfDay'] = order_df['OrderDate'].dt.hour
order_df['DayOfWeek'] = order_df['OrderDate'].dt.dayofweek
order_df['Month'] = order_df['OrderDate'].dt.month
order_df['DayOfMonth'] = order_df['OrderDate'].dt.day

# Discount
order_df['PromoCodeDiscount'] = order_df[['PromoCodeDiscount']].fillna(0).sum(axis=1)

# Revised shipping flags
order_df['ShippingType'] = np.select(
    [
        (order_df['OrderItemPriceTotal'] < 75) & (order_df['ShippingTotal'] == 0),         # Free but slow
        (order_df['OrderItemPriceTotal'] < 75) & (order_df['ShippingTotal'] == 4),         # Paid 2-day
        (order_df['OrderItemPriceTotal'] >= 75) & (order_df['ShippingTotal'] == 0),        # Free 2-day
        (order_df['ShippingTotal'] > 4)                                                    # Expedited
    ],
    [
        'NormalShipping',
        'TwoDayShipping',
        'TwoDayShipping',
        'ExpeditedShipping'
    ],
    default='UnknownShipping'
)

order_flags = (
    order_item_df
    .groupby('OrderNumber')[['IsSticker', 'IsLabel']]
    .any()  # True if any item in the order is a sticker/label
    .rename(columns={'IsSticker': 'OrderContainsSticker', 'IsLabel': 'OrderContainsLabel'})
    .reset_index()
)


# Average quantity per order
avg_quantity = order_item_df.groupby('OrderNumber')['Quantity'].mean().reset_index(name='AvgItemQuantity')
order_df = order_df.merge(avg_quantity, on='OrderNumber', how='left')

# Merge sticker/label flags into order_df
order_df = order_df.merge(order_flags, on='OrderNumber', how='left')
order_df['OrderContainsSticker'] = order_df['OrderContainsSticker'].fillna(False)
order_df['OrderContainsLabel'] = order_df['OrderContainsLabel'].fillna(False)


# -------------------------------------------------------------------------
# 2) TEMPORAL + ONE-HOT ENCODINGS
# -------------------------------------------------------------------------
def categorize_hour(row):
    if 5 <= row < 12:
        return 'Morning'
    elif 12 <= row < 17:
        return 'Afternoon'
    elif 17 <= row < 24:
        return 'Evening'
    else:
        return 'EarlyMorning'

order_df['HourCategory'] = order_df['HourOfDay'].apply(categorize_hour)
order_df['WorkHours'] = order_df.apply(lambda x: 8 <= x['HourOfDay'] <= 17 and x['DayOfWeek'] < 5, axis=1)
order_df['Workday'] = order_df['DayOfWeek'] < 5

order_df['FirstWeek'] = order_df['DayOfMonth'] <= 7
order_df['LastWeek'] = order_df['OrderDate'].apply(lambda d: d.day >= (d.days_in_month - 6))

order_df['Holiday'] = order_df['Month'].isin([11, 12])
order_df['Summer'] = order_df['Month'].isin([5, 6, 7, 8])
order_df['ElectionSeason'] = (order_df['Month'].isin([9, 10])) & (order_df['OrderDate'].dt.year % 2 == 0)
order_df['PresidentialElection'] = (order_df['Month'].isin([8, 9, 10])) & (order_df['OrderDate'].dt.year % 4 == 0)

order_df['RushFeeAndShipPrice'] = order_df['RushFee'] + order_df['ShipPrice']

# Save for future
order_df.to_feather("outputs/order_df_retention.feather")



# -------------------------------------------------------------------------
# 3) CUSTOMER-LEVEL AGGREGATES
# -------------------------------------------------------------------------

# Get each customer's relevant orders
if first_or_last == 'first':
    selected_orders = order_df.sort_values('OrderDate').groupby('CustomerId', as_index=False).head(number_of_orders)
elif first_or_last == 'last':
    selected_orders = order_df.sort_values('OrderDate').groupby('CustomerId', as_index=False).tail(number_of_orders)
else:
    selected_orders = order_df.copy()

grouped_customers = selected_orders.groupby('CustomerId')


def get_mode(x):
    return x.mode().iloc[0] if not x.mode().empty else np.nan

# Split up for speed
# agg_df = grouped_customers.agg(
#     TotalOrders=('OrderNumber', 'count'),
#     AvgOrderItemTotal=('OrderItemPriceTotal', 'mean'),
#     AvgPriceTotal=('PriceTotal', 'mean'),
#     AvgPromoDiscount=('PromoCodeDiscount', 'mean'),
#     AvgQuantity=('AvgItemQuantity', 'mean'),
#     AvgRushFee=('RushFee', 'mean'),
#     AvgShipPrice=('ShipPrice', 'mean'),
#     AvgRushFeeAndShipPrice=('RushFeeAndShipPrice', 'mean'),
#     ModeHour=('HourOfDay', get_mode),
#     ModeDayOfWeek=('DayOfWeek', get_mode),
#     ModeMonth=('Month', get_mode),
#     FirstOrderDate=('OrderDate', 'min'),
#     LastOrderDate=('OrderDate', 'max'),
#     MeanDaysBetweenOrders=('OrderDate', lambda x: x.sort_values().diff().dt.days.mean()),
#     StdDaysBetweenOrders=('OrderDate', lambda x: x.sort_values().diff().dt.days.std())
# ).reset_index()



agg_base = grouped_customers.agg(
    TotalOrders=('OrderNumber', 'count'),
    # For unit economics
    AvgOrderValue=('PriceTotal', 'mean'),
    MaxOrderValue=('PriceTotal', 'max'),
    MinOrderValue=('PriceTotal', 'min'),
    LifetimeValue=('PriceTotal', 'sum'),
    AvgItemsPerOrder=('ItemCount', 'mean'),
    AvgOrderItemTotal=('OrderItemPriceTotal', 'mean'),
    AvgPriceTotal=('PriceTotal', 'mean'),
    AvgPromoDiscount=('PromoCodeDiscount', 'mean'),
    AvgQuantity=('AvgItemQuantity', 'mean'),
    AvgRushFee=('RushFee', 'mean'),
    AvgShipPrice=('ShipPrice', 'mean'),
    AvgRushFeeAndShipPrice=('RushFeeAndShipPrice', 'mean'),
    FirstOrderDate=('OrderDate', 'min'),
    LastOrderDate=('OrderDate', 'max'),
).reset_index()


def custom_stats(group):
    order_dates = group['OrderDate'].sort_values()
    gaps = order_dates.diff().dt.days.dropna()
    return {
        'CustomerId': group['CustomerId'].iloc[0],
        'MeanDaysBetweenOrders': gaps.mean(),
        'StdDaysBetweenOrders': gaps.std(),
        'ModeHour': group['HourOfDay'].mode().iloc[0] if not group['HourOfDay'].mode().empty else np.nan,
        'ModeDayOfWeek': group['DayOfWeek'].mode().iloc[0] if not group['DayOfWeek'].mode().empty else np.nan,
        'ModeMonth': group['Month'].mode().iloc[0] if not group['Month'].mode().empty else np.nan
    }

# Filter to customers with specific ordres or more than 1
if first_or_last == 'first':
    grouped = order_df.sort_values('OrderDate').groupby('CustomerId', as_index=False).head(number_of_orders)
    filtered_groups = dict(tuple(grouped.groupby('CustomerId')))
elif first_or_last == 'last':
    grouped = order_df.sort_values('OrderDate').groupby('CustomerId', as_index=False).tail(number_of_orders)
    filtered_groups = dict(tuple(grouped.groupby('CustomerId')))
else:
    grouped = order_df.groupby('CustomerId')
    filtered_groups = {k: g for k, g in grouped if len(g) > 1}

agg_custom = pd.DataFrame(
    Parallel(n_jobs=8)(
        delayed(custom_stats)(group) for group in filtered_groups.values()
    )
)

agg_df = pd.merge(agg_base, agg_custom, on='CustomerId', how='left')

# agg_df.to_feather("outputs/customer_summary_first_order.feather")
# agg_df.to_feather("outputs/customer_summary_last_order.feather")



# Percentage encodings
def get_percentages(df, col, labels):
    total = df.groupby('CustomerId')[col].count()
    return pd.concat([
        df[df[col] == label].groupby('CustomerId')[col].count().div(total).rename(f'Pct{label}')
        for label in labels
    ], axis=1).fillna(0)

# Hour category %
hour_pct = get_percentages(order_df, 'HourCategory', ['Morning', 'Afternoon', 'Evening', 'EarlyMorning'])
work_pct = order_df.groupby('CustomerId')['WorkHours'].mean().rename('PctWorkHours')

# Day category %
day_pct = order_df.groupby('CustomerId')['Workday'].mean().rename('PctWorkday')

# Month categories
month_flags = ['Holiday', 'Summer', 'ElectionSeason', 'PresidentialElection']
month_pct = order_df.groupby('CustomerId')[month_flags].mean().add_prefix('Pct')

# Week of month %
week_flags = ['FirstWeek', 'LastWeek']
week_pct = order_df.groupby('CustomerId')[week_flags].mean().add_prefix('Pct')

# Shipping %
ship_pct = get_percentages(order_df, 'ShippingType', ['NormalShipping', 'TwoDayShipping', 'ExpeditedShipping'])

# Product type %
sticker_pct = order_df.groupby('CustomerId')['OrderContainsSticker'].mean().rename('PctOrdersWithSticker')
label_pct = order_df.groupby('CustomerId')['OrderContainsLabel'].mean().rename('PctOrdersWithLabel')


# -------------------------------------------------------------------------
# 4) PATTERN DETECTORS
# -------------------------------------------------------------------------

repeat_customers = agg_base['CustomerId'][agg_base['TotalOrders'] > 1]
order_df_no_singles = order_df[order_df['CustomerId'].isin(repeat_customers)]

def detect_repeating_pattern(group, months, min_years=2, min_gap_days=290, max_gap_days=2000):
    filtered = group[group['OrderDate'].dt.month.isin(months)]
    if len(filtered) < 2:
        return False
    years = filtered['OrderDate'].dt.year.nunique()
    max_gap = filtered['OrderDate'].sort_values().diff().dt.days.max()
    return (
        (years >= min_years)
        and (max_gap >= min_gap_days)
        and (max_gap <= max_gap_days) # ensures it happens within a cycle
    )


holiday_flag = (order_df_no_singles.groupby('CustomerId', group_keys=False)
    .apply(lambda g: detect_repeating_pattern(g, [11, 12], min_gap_days = 290), include_groups=False)
    .rename('HolidayPattern'))


summer_flag = (order_df_no_singles.groupby('CustomerId', group_keys=False)
               .apply(lambda g: detect_repeating_pattern(g, [5, 6, 7, 8], min_gap_days=220), include_groups=False)
               .rename('SummerPattern'))

election_flag = (
    order_df_no_singles[order_df_no_singles['ElectionSeason']]
    .groupby('CustomerId', group_keys=False)
    .apply(lambda g: detect_repeating_pattern(g, [9, 10], min_years=2, max_gap_days=800), include_groups=False)
    .rename('ElectionPattern')
)


presidential_flag = (order_df_no_singles[order_df_no_singles['PresidentialElection']].groupby('CustomerId', group_keys=False)
    .apply(lambda g: detect_repeating_pattern(g, [8, 9, 10], min_years=8, max_gap_days=1600), include_groups=False)
    .rename('PresidentialPattern'))


def detect_annual_pattern(group, min_pct=0.5, min_orders=2):
    order_dates = group['OrderDate'].sort_values()
    if len(order_dates) < min_orders:
        return False
    gaps = order_dates.diff().dt.days.dropna()
    pct_annual = (gaps.between(315, 415)).mean()  # fraction of gaps in the annual range
    return pct_annual >= min_pct

annual_flag = order_df_no_singles.groupby('CustomerId', group_keys=False).apply(
    detect_annual_pattern, include_groups=False
).rename('AnnualPattern')



# -------------------------------------------------------------------------
# 5) CHURN FLAGS
# -------------------------------------------------------------------------
today = order_df['OrderDate'].max()

churn_df = (
    agg_df
    .merge(holiday_flag, on='CustomerId', how='left')
    .merge(summer_flag, on='CustomerId', how='left')
    .merge(election_flag, on='CustomerId', how='left')
    .merge(presidential_flag, on='CustomerId', how='left')
    .merge(annual_flag, on='CustomerId', how='left')
)

churn_df['DaysSinceLastOrder'] = (today - churn_df['LastOrderDate']).dt.days

churn_df['NewCustomerLast90Days'] = churn_df['FirstOrderDate'] >= (today - timedelta(days=90))

churn_df['ExceededCadence'] = (
    churn_df['MeanDaysBetweenOrders'].notna() &                    # only customers with ≥2 orders
    (churn_df['DaysSinceLastOrder'] >
     1.5 * churn_df['MeanDaysBetweenOrders'])
)


churn_df['LikelyChurned'] = (
    (churn_df['ExceededCadence'] & churn_df['DaysSinceLastOrder'] > 35) |
    (
        (churn_df['DaysSinceLastOrder'] > 100) &
        ~(
            (churn_df['AnnualPattern'] & (churn_df['DaysSinceLastOrder'] <= 290)) |
            (churn_df['HolidayPattern'] & (churn_df['DaysSinceLastOrder'] <= 290)) |
            (churn_df['SummerPattern'] & (churn_df['DaysSinceLastOrder'] <= 220)) |
            (churn_df['ElectionPattern'] & (churn_df['DaysSinceLastOrder'] <= 700)) |
            (churn_df['PresidentialPattern'] & (churn_df['DaysSinceLastOrder'] <= 1300))
        )
    )
)


# This needs to go back in after the single-order people are added back
churn_df['OneAndDone'] = ((churn_df['TotalOrders'] == 1) & (churn_df['LikelyChurned'])).astype(int)

# This isn't necessarily accurate by the defintion of churn
# I wanted to keep OneAndDone separate from ThreeOrFewerAndChurned, i.e.
churn_df['ThreeOrFewerAndChurned'] = (churn_df['TotalOrders'] <= 3) & (churn_df['LikelyChurned'])
churn_df['TwoAndChurned'] = (churn_df['TotalOrders'] == 2) & (churn_df['LikelyChurned'])

# Pattern flags
# Old definitions, percentage based and not rule-based
# churn_df['QuarterlyPattern'] = churn_df['MeanDaysBetweenOrders'].between(75, 105)
# churn_df['MonthlyPattern'] = churn_df['MeanDaysBetweenOrders'].between(25, 35)
# churn_df['SeasonalPattern'] = month_pct['PctSummer'] > 0.5
# churn_df['ElectionPattern'] = month_pct['PctElectionSeason'] > 0.4
# churn_df['LikelyChurned'] = (
#     (churn_df['DaysSinceLastOrder'] > 90) &
#     ~(churn_df['QuarterlyPattern'] | churn_df['MonthlyPattern'] | churn_df['SeasonalPattern'] | churn_df['ElectionPattern'])
# )



# -------------------------------------------------------------------------
# MOMENTUM METRICS
# -------------------------------------------------------------------------

# Ensure First and Last order dates are datetime
churn_df['FirstOrderDate'] = pd.to_datetime(churn_df['FirstOrderDate'], errors='coerce')
churn_df['LastOrderDate'] = pd.to_datetime(churn_df['LastOrderDate'], errors='coerce')

# Compute duration of customer activity (avoid division by zero)
churn_df['CustomerDurationDays'] = (churn_df['LastOrderDate'] - churn_df['FirstOrderDate']).dt.days.clip(lower=1)

# Recent value rate (spend per day)
churn_df['RecentValueRate'] = churn_df['LifetimeValue'] / churn_df['CustomerDurationDays']

# Recent frequency (orders per day)
churn_df['RecentFrequency'] = churn_df['TotalOrders'] / churn_df['CustomerDurationDays']

# Handle any infinite or missing values
churn_df['RecentValueRate'] = churn_df['RecentValueRate'].replace([np.inf, -np.inf], np.nan).fillna(0)
churn_df['RecentFrequency'] = churn_df['RecentFrequency'].replace([np.inf, -np.inf], np.nan).fillna(0)



# -------------------------------------------------------------------------
# 6) FINAL MERGE
# -------------------------------------------------------------------------
customer_summary = (
    churn_df
    .merge(hour_pct, on='CustomerId', how='left')
    .merge(work_pct, on='CustomerId', how='left')
    .merge(day_pct, on='CustomerId', how='left')
    .merge(month_pct, on='CustomerId', how='left')
    .merge(week_pct, on='CustomerId', how='left')
    .merge(ship_pct, on='CustomerId', how='left')
    .merge(sticker_pct, on='CustomerId', how='left')
    .merge(label_pct, on='CustomerId', how='left')
)


# Save if needed
# customer_summary.to_csv("outputs/customer_summary.csv", index=False)
# customer_summary_subset = customer_summary.head(2000)
# customer_summary_subset.to_csv("outputs/customer_summary_subset.csv", index=False)
if first_or_last == None:
    filename = "outputs/customer_summary.feather"
else:
    filename = f"outputs/customer_summary_{first_or_last}_{number_of_orders}_orders.feather"

# Save
customer_summary.to_feather(filename)

gc.collect()