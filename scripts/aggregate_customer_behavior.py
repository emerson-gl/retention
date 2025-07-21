import pandas as pd
import numpy as np
import os, gc
from datetime import timedelta
from joblib import Parallel, delayed

# In order to run a "first_or_last" iteration of this script, you must first
# run it as "None" in order to save the relevant churn_save df to your outputs.
# This allows for the characterization of, say, OneAndDone to be appended
# to an aggregation of all customers' first orders.

# ───────────────────────── PARAMETERS ─────────────────────────
# first_or_last   = 'first'          # 'first'  'last'  or None
first_or_last   = None
number_of_orders= 1                # used only when ↑ is not None
os.chdir('C:/Users/Graphicsland/Spyder/retention')

# ───────────────────────── 1  LOAD + CLEAN ─────────────────────────
order_df = pd.read_feather('../sharedData/raw_order_df.feather')
order_df = (
    order_df
    [ (order_df['IsDeleted'] != 1) & (order_df['AffiliateId'] == 2) ]
    .dropna(subset=['ShippedDateTime'])
    .copy()
)

item_df  = pd.read_feather('../sharedData/raw_order_item_df.feather')
item_df = (
    item_df[(item_df['IsDeleted'] != 1)]
    .copy()
)
ppo      = pd.read_feather('../sharedData/raw_product_product_option_df.feather')

# For testing
# order_df['ShippedDateTime'] = pd.to_datetime(order_df['ShippedDateTime'], errors='coerce')
# order_df = order_df[
#     ((order_df['ShippedDateTime'] >= '2021-11-01') &
#     (order_df['ShippedDateTime'] < '2022-01-01')) |
#     ((order_df['ShippedDateTime'] >= '2022-11-01') &
#     (order_df['ShippedDateTime'] < '2023-01-01')) |
#     ((order_df['ShippedDateTime'] >= '2023-11-01') &
#     (order_df['ShippedDateTime'] < '2024-01-01')) 
# ]
# order_df = order_df[
#     ((order_df['ShippedDateTime'] >= '2021-11-01') &
#     (order_df['ShippedDateTime'] < '2022-01-01')) |
#     ((order_df['ShippedDateTime'] >= '2022-11-01') &
#     (order_df['ShippedDateTime'] < '2023-01-01')) |
#     ((order_df['ShippedDateTime'] >= '2023-11-01') &
#     (order_df['ShippedDateTime'] < '2024-01-01')) |
#     ((order_df['ShippedDateTime'] >= '2020-05-01') &
#     (order_df['ShippedDateTime'] < '2020-11-01')) |
#     ((order_df['ShippedDateTime'] >= '2022-05-01') &
#     (order_df['ShippedDateTime'] < '2022-11-01')) |
#     ((order_df['ShippedDateTime'] >= '2024-05-01') &
#     (order_df['ShippedDateTime'] < '2024-11-01'))
# ]


ppo['IsSticker'] = ppo['Name'].str.contains('icker',case=False,na=False) & ~ppo['Name'].str.contains('ample',case=False,na=False)
ppo['IsLabel']   = ppo['Name'].str.contains('abel' ,case=False,na=False) & ~ppo['Name'].str.contains('ample',case=False,na=False)
ppo['IsPouch']   = ppo['Name'].str.contains('ouch' ,case=False,na=False) & ~ppo['Name'].str.contains('ample',case=False,na=False)
item_df          = item_df.merge(ppo[['Id','IsSticker','IsLabel', 'IsPouch']],
                                 left_on='ProductProductOptionId', right_on='Id', how='left')



# basic per-order aggregates
counts   = item_df.groupby('OrderNumber').size().rename('ItemCount')
avg_qty  = item_df.groupby('OrderNumber')['Quantity'].mean().rename('AvgItemQuantity')
flags    = (item_df.groupby('OrderNumber')[['IsSticker','IsLabel', 'IsPouch']].any()
            .rename(columns={'IsSticker':'OrderContainsSticker','IsLabel':'OrderContainsLabel','IsPouch':'OrderContainsPouch'}))

order_df = (order_df
    .join(counts,     on='OrderNumber')
    .join(avg_qty,    on='OrderNumber')
    .join(flags,      on='OrderNumber')
    .fillna({'OrderContainsSticker':False,'OrderContainsLabel':False,'OrderContainsLabel':False}))

# datetime columns
order_df['PlacedDateTime']  = pd.to_datetime(order_df['PlacedDateTime'])
order_df['ShippedDateTime'] = pd.to_datetime(order_df['ShippedDateTime'])

order_df['HourOfDay']       = order_df['PlacedDateTime'].dt.hour
order_df['DayOfWeek']       = order_df['PlacedDateTime'].dt.dayofweek
order_df['Month']           = order_df['PlacedDateTime'].dt.month
order_df['DayOfMonth']      = order_df['PlacedDateTime'].dt.day
order_df['ShipDelayHours']  = (order_df['ShippedDateTime']-order_df['PlacedDateTime']).dt.total_seconds()/3600
order_df['PromoCodeDiscount']=order_df['PromoCodeDiscount'].fillna(0)

# shipping type
order_df['ShippingType'] = np.select(
    [(order_df['OrderItemPriceTotal']<75)&(order_df['ShippingTotal']==0),
     (order_df['OrderItemPriceTotal']<75)&(order_df['ShippingTotal']==4),
     (order_df['OrderItemPriceTotal']>=75)&(order_df['ShippingTotal']==0),
     (order_df['ShippingTotal']>4)],
    ['NormalShipping','TwoDayShipping','TwoDayShipping','ExpeditedShipping'],
    default='UnknownShipping')

# temporal flags
order_df['HourCategory']     = order_df['HourOfDay'].apply(
    lambda h:'Morning' if 5<=h<12 else 'Afternoon' if 12<=h<17 else 'Evening' if 17<=h<24 else 'EarlyMorning')
order_df['WorkHours']        = (order_df['HourOfDay'].between(8,17)) & (order_df['DayOfWeek']<5)
order_df['Workday']          = order_df['DayOfWeek']<5
order_df['FirstWeek']        = order_df['DayOfMonth']<=7
order_df['LastWeek']         = order_df['PlacedDateTime'].dt.days_in_month-order_df['DayOfMonth']<7
order_df['Holiday']          = order_df['Month'].isin([11,12])
order_df['Summer']           = order_df['Month'].isin([5,6,7,8])
order_df['ElectionSeason']   = order_df['Month'].isin([9,10]) & (order_df['PlacedDateTime'].dt.year%2==0)
order_df['PresidentialElection']= order_df['Month'].isin([8,9,10]) & (order_df['PlacedDateTime'].dt.year%4==0)
order_df['RushFeeAndShipPrice']= order_df['RushFee']+order_df['ShipPrice']

order_df.to_feather('outputs/order_df_retention.feather')

today = order_df['PlacedDateTime'].max()

# ───────────────────────── 2  CUSTOMER-LEVEL AGG ─────────────────────────
def subset_orders(df):
    if first_or_last=='first':
        return df.sort_values('PlacedDateTime').groupby('CustomerId').head(number_of_orders)
    if first_or_last=='last':
        return df.sort_values('PlacedDateTime').groupby('CustomerId').tail(number_of_orders)
    return df
sel = subset_orders(order_df)

def mode_or_nan(s): return s.mode().iloc[0] if not s.mode().empty else np.nan
agg_df = (sel.groupby('CustomerId')
    .agg(TotalOrders=('OrderNumber','count'),
         AvgOrderItemTotal=('OrderItemPriceTotal','mean'),
         AvgPriceTotal=('PriceTotal','mean'),
         AvgItemsPerOrder=('ItemCount','mean'),
         AvgPromoDiscount=('PromoCodeDiscount','mean'),
         AvgQuantity=('AvgItemQuantity','mean'),
         AvgRushFee=('RushFee','mean'),
         AvgShipPrice=('ShipPrice','mean'),
         AvgRushFeeAndShipPrice=('RushFeeAndShipPrice','mean'),
         FirstOrderDate=('PlacedDateTime','min'),
         LastOrderDate=('PlacedDateTime','max'),
         MeanDaysBetweenOrders=('PlacedDateTime',lambda x:x.sort_values().diff().dt.days.mean()),
         StdDaysBetweenOrders=('PlacedDateTime',lambda x:x.sort_values().diff().dt.days.std()),
         ModeHour=('HourOfDay',mode_or_nan),
         ModeDayOfWeek=('DayOfWeek',mode_or_nan),
         ModeMonth=('Month',mode_or_nan))
    .reset_index())

# ───────────────────────── 3  PERCENTAGE ENCODINGS ─────────────────────────
def pct(df, col, labels):
    tot = df.groupby('CustomerId')['OrderNumber'].nunique()
    pieces = [
        df[df[col] == l].groupby('CustomerId')['OrderNumber'].nunique().div(tot).rename(f'Pct{l}')
        for l in labels
    ]
    return pd.concat(pieces, axis=1).fillna(0)


ship_pct   = pct(order_df,'ShippingType',['NormalShipping','TwoDayShipping','ExpeditedShipping'])
hour_pct   = pct(order_df,'HourCategory',['Morning','Afternoon','Evening','EarlyMorning'])
month_pct  = order_df.groupby('CustomerId')[['Holiday','Summer','ElectionSeason','PresidentialElection']].mean().add_prefix('Pct')
week_pct   = order_df.groupby('CustomerId')[['FirstWeek','LastWeek']].mean().add_prefix('Pct')

add_pct = pd.concat([
    ship_pct, hour_pct, week_pct, month_pct, 
    order_df.groupby('CustomerId')['WorkHours'].mean().rename('PctWorkHours'),
    order_df.groupby('CustomerId')['Workday'  ].mean().rename('PctWorkday'),
    order_df.groupby('CustomerId')['OrderContainsSticker'].mean().rename('PctOrdersWithSticker'),
    order_df.groupby('CustomerId')['OrderContainsLabel'  ].mean().rename('PctOrdersWithLabel'),
    order_df.groupby('CustomerId')['OrderContainsPouch'  ].mean().rename('PctOrdersWithPouch')
],axis=1).reset_index()


# ─────────── 4  PATTERN DETECTORS ───────────  

if first_or_last is None:
    repeat_customers = agg_df.loc[agg_df['TotalOrders'] > 1, 'CustomerId']  
    order_df_no_singles = order_df[order_df['CustomerId'].isin(repeat_customers)]
    
    
    
    def detect_repeating_pattern(group, months, min_years=2, min_gap_days=290, max_gap_days=2000):
        filtered = group[group['PlacedDateTime'].dt.month.isin(months)]
        if len(filtered) < 2:
            return False
        years = filtered['PlacedDateTime'].dt.year.nunique()
        max_gap = filtered['PlacedDateTime'].sort_values().diff().dt.days.max()
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
        order_dates = group['PlacedDateTime'].sort_values()
        if len(order_dates) < min_orders:
            return False
        gaps = order_dates.diff().dt.days.dropna()
        pct_annual = (gaps.between(315, 415)).mean()  # fraction of gaps in the annual range
        return pct_annual >= min_pct
    
    annual_flag = order_df_no_singles.groupby('CustomerId', group_keys=False).apply(
        detect_annual_pattern, include_groups=False
    ).rename('AnnualPattern')
    
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
    
    
    
    recently_active = churn_df['DaysSinceLastOrder'] <= 35
    
    rule_cadence = (
        churn_df['ExceededCadence'] &
        (churn_df['DaysSinceLastOrder'] > 35)
    )
    
    rule_pattern = (
        (churn_df['DaysSinceLastOrder'] > 100) &
        ~(
            (churn_df['AnnualPattern']       & (churn_df['DaysSinceLastOrder'] <= 290)) |
            (churn_df['HolidayPattern']      & (churn_df['DaysSinceLastOrder'] <= 290)) |
            (churn_df['SummerPattern']       & (churn_df['DaysSinceLastOrder'] <= 220)) |
            (churn_df['ElectionPattern']     & (churn_df['DaysSinceLastOrder'] <= 700)) |
            (churn_df['PresidentialPattern'] & (churn_df['DaysSinceLastOrder'] <= 1300))
        )
    )
    
    churn_df['LikelyChurned'] = ~recently_active & (rule_cadence | rule_pattern)
    
    # This needs to go back in after the single-order people are added back
    churn_df['OneAndDone'] = ((churn_df['TotalOrders'] == 1) & (churn_df['LikelyChurned']))
    
    # This isn't necessarily accurate by the defintion of churn
    # I wanted to keep OneAndDone separate from ThreeOrFewerAndChurned, i.e.
    churn_df['ThreeOrFewerAndChurned'] = (churn_df['TotalOrders'] <= 3) & (churn_df['LikelyChurned'])
    churn_df['TwoAndChurned'] = (churn_df['TotalOrders'] == 2) & (churn_df['LikelyChurned'])
    
    churn_save = churn_df[['CustomerId', 'DaysSinceLastOrder', 'NewCustomerLast90Days', 'ExceededCadence', 
                           'AnnualPattern', 'HolidayPattern', 'SummerPattern', 'ElectionPattern', 'PresidentialPattern',
                           'LikelyChurned', 'OneAndDone', 'ThreeOrFewerAndChurned', 'TwoAndChurned']]
    churn_save.to_feather('outputs/churn_save.feather')
    
    customer_summary = churn_df.merge(add_pct, on = 'CustomerId', how='left')
    
    customer_summary.to_feather('outputs/customer_summary.feather')

    
else:
    
    churn_df = agg_df
    
    churn_load = pd.read_feather('outputs/churn_save.feather')
    
    churn_df = churn_df.merge(churn_load, on = 'CustomerId', how = 'left')

    customer_summary = churn_df.merge(add_pct, on = 'CustomerId', how='left')
    
    if number_of_orders == 1:
        customer_summary.to_feather(f"outputs/customer_summary_{first_or_last}_order.feather")
    else:
        customer_summary.to_feather(f"outputs/customer_summary_{first_or_last}_{number_of_orders}_orders.feather")


gc.collect()
