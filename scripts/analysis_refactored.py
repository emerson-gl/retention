import pandas as pd
import numpy as np
import os, gc
from datetime import timedelta
from joblib import Parallel, delayed

# ───────────────────────── PARAMETERS ─────────────────────────
first_or_last   = 'first'          # 'first'  'last'  or None
number_of_orders= 1                # used only when ↑ is not None
os.chdir('C:/Users/Graphicsland/Spyder/retention')

# ───────────────────────── 1  LOAD + CLEAN ─────────────────────────
order_df = pd.read_feather('../sharedData/raw_order_df.feather')
item_df  = pd.read_feather('../sharedData/raw_order_item_df.feather')
ppo      = pd.read_feather('../sharedData/raw_product_product_option_df.feather')

ppo['IsSticker'] = ppo['Name'].str.contains('icker',case=False,na=False) & ~ppo['Name'].str.contains('ample',case=False,na=False)
ppo['IsLabel']   = ppo['Name'].str.contains('abel' ,case=False,na=False) & ~ppo['Name'].str.contains('ample',case=False,na=False)
item_df          = item_df.merge(ppo[['Id','IsSticker','IsLabel']],
                                 left_on='ProductProductOptionId', right_on='Id', how='left')

order_df = (order_df
    .query('IsDeleted!=True and AffiliateId==2')
    .dropna(subset=['ShippedDateTime'])
    .copy())

# basic per-order aggregates
counts   = item_df.groupby('OrderNumber').size().rename('ItemCount')
avg_qty  = item_df.groupby('OrderNumber')['Quantity'].mean().rename('AvgItemQuantity')
flags    = (item_df.groupby('OrderNumber')[['IsSticker','IsLabel']].any()
            .rename(columns={'IsSticker':'OrderContainsSticker','IsLabel':'OrderContainsLabel'}))

order_df = (order_df
    .join(counts,     on='OrderNumber')
    .join(avg_qty,    on='OrderNumber')
    .join(flags,      on='OrderNumber')
    .fillna({'OrderContainsSticker':False,'OrderContainsLabel':False}))

# datetime columns
order_df['OrderDate']       = pd.to_datetime(order_df['OrderDate'],      errors='coerce')
order_df['PlacedDateTime']  = pd.to_datetime(order_df['PlacedDateTime'])
order_df['ShippedDateTime'] = pd.to_datetime(order_df['ShippedDateTime'])

order_df['HourOfDay']       = order_df['OrderDate'].dt.hour
order_df['DayOfWeek']       = order_df['OrderDate'].dt.dayofweek
order_df['Month']           = order_df['OrderDate'].dt.month
order_df['DayOfMonth']      = order_df['OrderDate'].dt.day
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
order_df['LastWeek']         = order_df['OrderDate'].dt.days_in_month-order_df['DayOfMonth']<7
order_df['Holiday']          = order_df['Month'].isin([11,12])
order_df['Summer']           = order_df['Month'].isin([5,6,7,8])
order_df['ElectionSeason']   = order_df['Month'].isin([9,10]) & (order_df['OrderDate'].dt.year%2==0)
order_df['PresidentialElection']= order_df['Month'].isin([8,9,10]) & (order_df['OrderDate'].dt.year%4==0)
order_df['RushFeeAndShipPrice']= order_df['RushFee']+order_df['ShipPrice']

order_df.to_feather('outputs/order_df_retention.feather')

# ───────────────────────── 2  CUSTOMER-LEVEL AGG ─────────────────────────
def subset_orders(df):
    if first_or_last=='first':
        return df.sort_values('OrderDate').groupby('CustomerId').head(number_of_orders)
    if first_or_last=='last':
        return df.sort_values('OrderDate').groupby('CustomerId').tail(number_of_orders)
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
         FirstOrderDate=('OrderDate','min'),
         LastOrderDate=('OrderDate','max'),
         MeanDaysBetweenOrders=('OrderDate',lambda x:x.sort_values().diff().dt.days.mean()),
         StdDaysBetweenOrders=('OrderDate',lambda x:x.sort_values().diff().dt.days.std()),
         ModeHour=('HourOfDay',mode_or_nan),
         ModeDayOfWeek=('DayOfWeek',mode_or_nan),
         ModeMonth=('Month',mode_or_nan))
    .reset_index())

# ───────────────────────── 3  PERCENTAGE ENCODINGS ─────────────────────────
def pct(df,col,labels):
    tot = df.groupby('CustomerId')[col].count()
    pieces=[df[df[col]==l].groupby('CustomerId')[col].count().div(tot).rename(f'Pct{l}') for l in labels]
    return pd.concat(pieces,axis=1).fillna(0)

hour_pct   = pct(order_df,'HourCategory',['Morning','Afternoon','Evening','EarlyMorning'])
ship_pct   = pct(order_df,'ShippingType',['NormalShipping','TwoDayShipping','ExpeditedShipping'])
month_pct  = order_df.groupby('CustomerId')[['Holiday','Summer','ElectionSeason','PresidentialElection']].mean().add_prefix('Pct')
week_pct   = order_df.groupby('CustomerId')[['FirstWeek','LastWeek']].mean().add_prefix('Pct')

add_pct = pd.concat([
    hour_pct,
    order_df.groupby('CustomerId')['WorkHours'].mean().rename('PctWorkHours'),
    order_df.groupby('CustomerId')['Workday'  ].mean().rename('PctWorkday'),
    month_pct, week_pct, ship_pct,
    order_df.groupby('CustomerId')['OrderContainsSticker'].mean().rename('PctOrdersWithSticker'),
    order_df.groupby('CustomerId')['OrderContainsLabel'  ].mean().rename('PctOrdersWithLabel')
],axis=1).reset_index()

# ─────────── 4  PATTERN DETECTORS ───────────  
rep_ids   = agg_df.loc[agg_df['TotalOrders'] > 1, 'CustomerId']  
repeat_df = order_df[order_df['CustomerId'].isin(rep_ids)]  

def detect_pattern(df, months, *, min_yrs=2, min_gap=290, max_gap=2000):  
    df = df[df['OrderDate'].dt.month.isin(months)]  
    if len(df) < 2:                       # need ≥ 2 orders  
        return False  
    yrs = df['OrderDate'].dt.year.nunique()  
    gap = df['OrderDate'].sort_values().diff().dt.days.max()  
    return (yrs >= min_yrs) and (min_gap <= gap <= max_gap)  

def detect_annual(df, *, min_pct=.5, min_orders=2):  
    d = df['OrderDate'].sort_values()  
    if len(d) < min_orders:  
        return False  
    gaps = d.diff().dt.days.dropna()  
    return (gaps.between(315, 415)).mean() >= min_pct  

patt = {  
    'HolidayPattern'     : repeat_df.groupby('CustomerId')                              .apply(lambda g: detect_pattern(g,[11,12],               min_gap=290)),  
    'SummerPattern'      : repeat_df.groupby('CustomerId')                              .apply(lambda g: detect_pattern(g,[5,6,7,8],             min_gap=220)),  
    'ElectionPattern'    : repeat_df[repeat_df['ElectionSeason'] ]                     .groupby('CustomerId') .apply(lambda g: detect_pattern(g,[9,10],min_yrs=2 ,max_gap=800)),  
    'PresidentialPattern': repeat_df[repeat_df['PresidentialElection']]                .groupby('CustomerId') .apply(lambda g: detect_pattern(g,[8,9,10],min_yrs=8 ,max_gap=1600)),  
    'AnnualPattern'      : repeat_df.groupby('CustomerId').apply(detect_annual)  
}  

#  make every pattern Series align to the master CustomerId list;  
#  this prevents the “all-scalar / different levels” errors.  
id_index = agg_df['CustomerId']                     # 1-D Index we trust  
pat_df   = pd.DataFrame({'CustomerId': id_index})   # start with the id column  
for name, ser in patt.items():  
    pat_df[name] = ser.reindex(id_index).fillna(False).values  

# ─────────── 5  CHURN + MOMENTUM ───────────  
today  = order_df['OrderDate'].max()  
churn  = agg_df.merge(pat_df, on='CustomerId', how='left').fillna(False)  

churn['DaysSinceLastOrder']    = (today - churn['LastOrderDate']).dt.days  
churn['NewCustomerLast90Days'] = churn['FirstOrderDate'] >= today - timedelta(days=90)  
churn['ExceededCadence']       = churn['MeanDaysBetweenOrders'].notna() & (churn['DaysSinceLastOrder'] > 1.5 * churn['MeanDaysBetweenOrders'])  

churn['LikelyChurned'] = (  
      churn['ExceededCadence'] & (churn['DaysSinceLastOrder'] > 35)  
) | (  
    (churn['DaysSinceLastOrder'] > 100) &  
    ~(  churn['AnnualPattern']       & (churn['DaysSinceLastOrder'] <= 290) |  
       churn['HolidayPattern']       & (churn['DaysSinceLastOrder'] <= 290) |  
       churn['SummerPattern']        & (churn['DaysSinceLastOrder'] <= 220) |  
       churn['ElectionPattern']      & (churn['DaysSinceLastOrder'] <= 700) |  
       churn['PresidentialPattern']  & (churn['DaysSinceLastOrder'] <= 1300))  
)  

churn['OneAndDone']             = ((churn['TotalOrders'] == 1) & churn['LikelyChurned']).astype(int)  
churn['ThreeOrFewerAndChurned'] = (churn['TotalOrders'] <= 3)  & churn['LikelyChurned']  
churn['TwoAndChurned']          = (churn['TotalOrders'] == 2)  & churn['LikelyChurned']  

churn['CustomerDurationDays'] = (churn['LastOrderDate'] - churn['FirstOrderDate']).dt.days.clip(lower=1)  
churn['RecentValueRate']      = churn['LifetimeValue'] / churn['CustomerDurationDays']  
churn['RecentFrequency']      = churn['TotalOrders']   / churn['CustomerDurationDays']  
churn.replace([np.inf, -np.inf], 0, inplace=True)  

# ─────────── 6  FINAL MERGE & EXPORT ───────────  
customer_summary = churn.merge(add_pct, on='CustomerId', how='left')  
customer_summary.to_feather('outputs/customer_summary.feather')  

first_orders = (order_df.sort_values('OrderDate')  
                .drop_duplicates('CustomerId')  
                .merge(customer_summary[['CustomerId','LikelyChurned','OneAndDone']],  
                       on='CustomerId', how='left'))  
first_orders.to_feather('outputs/customer_summary_first_order.feather')  

gc.collect()
