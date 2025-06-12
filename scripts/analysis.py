import pandas as pd
import numpy as np
import os
from datetime import timedelta

# -------------------------------------------------------------------------
# 1) LOAD AND CLEAN
# -------------------------------------------------------------------------
os.chdir('C:\\Users\\Graphicsland\\Spyder\\retention')

# Load data
order_df = pd.read_feather("C:/Users/Graphicsland/Spyder/sharedData/raw_order_df.feather")
order_item_df = pd.read_feather("C:/Users/Graphicsland/Spyder/sharedData/raw_order_item_df.feather")

# Clean
order_df = order_df[(order_df['IsDeleted'] != True) & (order_df['AffiliateId'] == 2)].copy()
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
order_df['TotalDiscount'] = order_df[['PromoCodeDiscount', 'TaxableDiscountAmount', 'NonTaxableDiscountAmount']].fillna(0).sum(axis=1)

# Shipping type mapping
shipping_map = {
    'FedExStandardOvernight': 'Rushed', 'FedExPriorityOvernight': 'Rushed', 'FedExFirstOvernight': 'Rushed',
    'FedExSaturdayDelivery': 'Rushed', 'UPSNextDayAir': 'Rushed', 'UPSNextDayAirSaver': 'Rushed',
    'UPSNextDayAirEarly': 'Rushed', 'UPSSaturdayDelivery': 'Rushed', 'MessengerService': 'Rushed',
    'ConciergeService': 'Rushed', 'FedEx2Day': 'Expedited', 'FedExOneRate2Day': 'Expedited',
    'UPS2DayAir': 'Expedited', 'FedExExpressSaver': 'Expedited', 'UPS3DaySelect': 'Expedited',
    'FedExInternationalPriority': 'Expedited', 'FedExInternationalConnectPlus': 'Expedited',
    'PickupAtConferenceExpedited': 'Expedited', 'MiExpedited': 'Expedited', 'FirstClassMail': 'Normal',
    'PriorityMail': 'Normal', 'USMail': 'Normal', 'UPSGround': 'Normal', 'FedExGroundHomeDelivery': 'Normal',
    'FedExGround': 'Normal', 'FedExInternationalGround': 'Normal', 'PriorityMailInternational': 'Normal',
    'FirstClassMailInternational': 'Normal', 'Standard': 'Normal', 'Other': 'Normal', 'Pickup': 'No-Ship',
    'PickupAtConference': 'No-Ship', 'PickupAtConferenceFree': 'No-Ship'
}
order_df['ShippingType'] = order_df['VendorShippingServiceID'].map(shipping_map).fillna('Unknown')

# Average quantity per order
avg_quantity = order_item_df.groupby('OrderNumber')['Quantity'].mean().reset_index(name='AvgItemQuantity')
order_df = order_df.merge(avg_quantity, on='OrderNumber', how='left')

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

grouped_customers = order_df.groupby('CustomerId')


def get_mode(x):
    return x.mode().iloc[0] if not x.mode().empty else np.nan

agg_df = grouped_customers.agg(
    TotalOrders=('OrderNumber', 'count'),
    AvgOrderItemTotal=('OrderItemPriceTotal', 'mean'),
    AvgPriceTotal=('PriceTotal', 'mean'),
    AvgDiscount=('TotalDiscount', 'mean'),
    AvgQuantity=('AvgItemQuantity', 'mean'),
    AvgRushFee=('RushFee', 'mean'),
    AvgShipPrice=('ShipPrice', 'mean'),
    AvgRushFeeAndShipPrice=('RushFeeAndShipPrice', 'mean'),
    ModeHour=('HourOfDay', get_mode),
    ModeDayOfWeek=('DayOfWeek', get_mode),
    ModeMonth=('Month', get_mode),
    FirstOrderDate=('OrderDate', 'min'),
    LastOrderDate=('OrderDate', 'max'),
    MeanDaysBetweenOrders=('OrderDate', lambda x: x.sort_values().diff().dt.days.mean()),
    StdDaysBetweenOrders=('OrderDate', lambda x: x.sort_values().diff().dt.days.std())
).reset_index()

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
ship_pct = get_percentages(order_df, 'ShippingType', ['Rushed', 'Expedited', 'Normal', 'No-Ship'])

# -------------------------------------------------------------------------
# 4) CHURN FLAGS
# -------------------------------------------------------------------------
today = order_df['OrderDate'].max()
churn_df = agg_df.copy()
churn_df['DaysSinceLastOrder'] = (today - churn_df['LastOrderDate']).dt.days
churn_df['OneAndDone'] = churn_df['TotalOrders'] == 1
churn_df['ThreeOrFewerAndChurned'] = (churn_df['TotalOrders'] <= 3) & (churn_df['DaysSinceLastOrder'] > 90)
churn_df['NewCustomerLast90Days'] = churn_df['FirstOrderDate'] >= (today - timedelta(days=90))

# Pattern flags
churn_df['QuarterlyPattern'] = churn_df['MeanDaysBetweenOrders'].between(75, 105)
churn_df['MonthlyPattern'] = churn_df['MeanDaysBetweenOrders'].between(25, 35)
churn_df['SeasonalPattern'] = month_pct['PctSummer'] > 0.5
churn_df['ElectionPattern'] = month_pct['PctElectionSeason'] > 0.4

churn_df['LikelyChurned'] = (
    (churn_df['DaysSinceLastOrder'] > 90) &
    ~(churn_df['QuarterlyPattern'] | churn_df['MonthlyPattern'] | churn_df['SeasonalPattern'] | churn_df['ElectionPattern'])
)

# -------------------------------------------------------------------------
# 5) FINAL MERGE
# -------------------------------------------------------------------------
customer_summary = (
    churn_df
    .merge(hour_pct, on='CustomerId', how='left')
    .merge(work_pct, on='CustomerId', how='left')
    .merge(day_pct, on='CustomerId', how='left')
    .merge(month_pct, on='CustomerId', how='left')
    .merge(week_pct, on='CustomerId', how='left')
    .merge(ship_pct, on='CustomerId', how='left')
)

# Save if needed
# customer_summary.to_csv("outputs/customer_summary.csv", index=False)
# customer_summary_subset = customer_summary.head(2000)
# customer_summary_subset.to_csv("outputs/customer_summary_subset.csv", index=False)
customer_summary.to_feather("outputs/customer_summary.feather")
