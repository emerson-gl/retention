import pandas as pd
import numpy as np
import os
from datetime import timedelta


os.chdir('C:\\Users\\Graphicsland\\Spyder\\retention')

# Load data
order_df = pd.read_feather("C:/Users/Graphicsland/Spyder/sharedData/raw_order_df.feather")
order_item_df = pd.read_feather("C:/Users/Graphicsland/Spyder/sharedData/raw_order_item_df.feather")

# Filter valid records
order_df = order_df[(order_df['IsDeleted'] != True) & (order_df['AffiliateId'] == 2)].copy()
order_item_df = order_item_df[order_item_df['IsDeleted'] != True].copy()

# Parse dates
order_df['OrderDate'] = pd.to_datetime(order_df['OrderDate'], errors='coerce')
order_df['PlacedDateTime'] = pd.to_datetime(order_df['PlacedDateTime'], errors='coerce')
order_df['ShippedDateTime'] = pd.to_datetime(order_df['ShippedDateTime'], errors='coerce')

# Enrich order_df
order_df['HourOfDay'] = order_df['OrderDate'].dt.hour
order_df['DayOfWeek'] = order_df['OrderDate'].dt.dayofweek
order_df['Month'] = order_df['OrderDate'].dt.month
order_df['DayOfMonth'] = order_df['OrderDate'].dt.day

order_df['TotalDiscount'] = order_df[['PromoCodeDiscount', 'TaxableDiscountAmount', 'NonTaxableDiscountAmount']].fillna(0).sum(axis=1)
order_df['RushFeeAndShipPrice'] = order_df[['RushFee', 'ShipPrice']].fillna(0).sum(axis=1)

# Shipping type
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

# Add seasonal flags
order_df['Holiday'] = order_df['Month'].isin([11, 12])
order_df['Summer'] = order_df['Month'].isin([5, 6, 7, 8])
order_df['ElectionSeason'] = order_df['Month'].isin([9, 10]) & (order_df['OrderDate'].dt.year % 2 == 0)
order_df['PresidentialElection'] = order_df['Month'].isin([8, 9, 10]) & (order_df['OrderDate'].dt.year % 4 == 0)

# Average item quantity per order
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


# Get each customer's last order
grouped_customers = order_df.sort_values('OrderDate').groupby('CustomerId', as_index=False).tail(1)

# Export
# last_orders.to_feather("outputs/customer_summary_one_order.feather")



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



# Percentage encodings
def get_percentages(df, col, labels):
    total = df.groupby('CustomerId')[col].count()
    return pd.concat([
        df[df[col] == label].groupby('CustomerId')[col].count().div(total).rename(f'Pct{label}')
        for label in labels
    ], axis=1).fillna(0)

# Hour category %
hour_pct = get_percentages(grouped_customers, 'HourCategory', ['Morning', 'Afternoon', 'Evening', 'EarlyMorning'])
work_pct = grouped_customers.groupby('CustomerId')['WorkHours'].mean().rename('PctWorkHours')

# Day category %
day_pct = grouped_customers.groupby('CustomerId')['Workday'].mean().rename('PctWorkday')

# Month categories
month_flags = ['Holiday', 'Summer', 'ElectionSeason', 'PresidentialElection']
month_pct = grouped_customers.groupby('CustomerId')[month_flags].mean().add_prefix('Pct')

# Week of month %
week_flags = ['FirstWeek', 'LastWeek']
week_pct = grouped_customers.groupby('CustomerId')[week_flags].mean().add_prefix('Pct')

# Shipping %
ship_pct = get_percentages(grouped_customers, 'ShippingType', ['Rushed', 'Expedited', 'NormalShipping', 'No-Ship'])


today = order_df['OrderDate'].max()


grouped_customers['DaysSinceLastOrder'] = (today - grouped_customers['PlacedDateTime']).dt.days

grouped_customers['NewCustomerLast90Days'] = grouped_customers['PlacedDateTime'] >= (today - timedelta(days=90))


customer_summary = pd.read_feather("C:/Users/Graphicsland/Spyder/retention/outputs/customer_summary.feather")
grouped_customers = grouped_customers.merge(customer_summary[['CustomerId', 'LikelyChurned', 'OneAndDone', 'ThreeOrFewerAndChurned']], on = 'CustomerId', how='left')

# Add ChurnedOrOneAndDone
grouped_customers['ChurnedOrOneAndDone'] = 0
grouped_customers.loc[grouped_customers['OneAndDone'] == 1, 'ChurnedOrOneAndDone'] = 1
grouped_customers.loc[(grouped_customers['LikelyChurned'] == 1) & (grouped_customers['OneAndDone'] != 1), 'ChurnedOrOneAndDone'] = 2

# Add ChurnedOrThreeAndDone
grouped_customers['ChurnedOrThreeAndDone'] = 0
grouped_customers.loc[grouped_customers['ThreeOrFewerAndChurned'] == 1, 'ChurnedOrThreeAndDone'] = 1
grouped_customers.loc[(grouped_customers['LikelyChurned'] == 1) & (grouped_customers['ThreeOrFewerAndChurned'] != 1), 'ChurnedOrThreeAndDone'] = 2



# -------------------------------------------------------------------------
# 6) FINAL MERGE
# -------------------------------------------------------------------------
customer_summary_one_order = (
    grouped_customers
    .merge(hour_pct, on='CustomerId', how='left')
    .merge(work_pct, on='CustomerId', how='left')
    .merge(day_pct, on='CustomerId', how='left')
    .merge(month_pct, on='CustomerId', how='left')
    .merge(week_pct, on='CustomerId', how='left')
    .merge(ship_pct, on='CustomerId', how='left')
)


# Save
customer_summary_one_order.to_feather('outputs/customer_summary_one_order.feather')

