import pandas as pd
import numpy as np
from xgboost import XGBClassifier, plot_importance
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, roc_auc_score
import matplotlib.pyplot as plt
import shap
import gc

# predict_variable = 'LikelyChurned'
predict_variable = 'TwoAndChurned'
# predict_variable = 'OneAndDone'

min_orders = 2

# Load your data

if predict_variable == 'LikelyChurned':
    customer_summary = pd.read_feather("C:/Users/Graphicsland/Spyder/retention/outputs/customer_summary.feather")
    customer_summary['Rolling90DaysSinceLastOrder'] = customer_summary['DaysSinceLastOrder'].clip(upper=90)
    customer_summary['Rolling30DaysSinceLastOrder'] = customer_summary['DaysSinceLastOrder'].clip(upper=30)

    
    # LikelyChurned
    # attributes = ['TotalOrders', 'AvgOrderItemTotal', 'AvgPriceTotal',
    #        'AvgDiscount', 'AvgQuantity', 'ModeHour', 'ModeDayOfWeek',
    #        'PctMorning', 'PctAfternoon', 'PctEvening', 'PctEarlyMorning',
    #        'PctWorkHours', 'PctWorkday', 'PctHoliday', 'PctSummer',
    #        'PctElectionSeason', 'PctPresidentialElection', 'PctFirstWeek',
    #        'PctLastWeek', 'PctRushed', 'PctExpedited', 'PctNormalShipping', 'PctNo-Ship',
    #        # 'HolidayPattern',
    #        # 'SummerPattern',
    #        # 'ElectionPattern',
    #        # 'PresidentialPattern',
    #        # 'AnnualPattern',
    # ]
    
    attributes = ['TotalOrders', 'AvgOrderItemTotal',
           'AvgDiscount', 
           # 'AvgQuantity',
           # 'PctHoliday', 
           # 'PctSummer',
           # 'PctElectionSeason', 
           # 'PctPresidentialElection',
           'PctNormalShipping',
           # 'HolidayPattern',
           # 'SummerPattern',
           # 'ElectionPattern',
           # 'PresidentialPattern',
           # 'AnnualPattern',
           # 'Rolling90DaysSinceLastOrder',
           # 'Rolling30DaysSinceLastOrder',
           'FirstOrderSampleOnly',
    ]
    
    # attributes = [
    #     'PctHoliday', 'AvgOrderItemTotal', 'AvgDiscount', 'PctWorkHours', 
    #     'PctSummer', 'PctPresidentialElection', 'PctNormalShipping', 
    # ]
    
elif predict_variable == 'TwoAndChurned':
    
    customer_summary = pd.read_feather("C:/Users/Graphicsland/Spyder/retention/outputs/customer_summary_first_2_order.feather")
    # total_df = customer_summary = pd.read_feather("C:/Users/Graphicsland/Spyder/retention/outputs/customer_summary.feather")
    # total_df = total_df[['CustomerId', 'TotalOrders']]
    # customer_summary = customer_summary.merge(total_df, on = 'CustomerId', how = 'left')
            
    # LikelyChurned
    attributes = ['AvgOrderItemTotal',
           'AvgDiscount', 
           # 'AvgQuantity',
           'PctHoliday', 
           # 'PctSummer',
           # 'PctElectionSeason', 
           # 'PctPresidentialElection',
           'PctNormalShipping',
           # 'HolidayPattern',
           # 'SummerPattern',
           # 'ElectionPattern',
           # 'PresidentialPattern',
           # 'AnnualPattern',
           # 'Rolling90DaysSinceLastOrder',s
           # 'Rolling30DaysSinceLastOrder',
           'FirstOrderSampleOnly',
    ]
    
    
    
elif predict_variable == 'OneAndDone':
    
    customer_summary = pd.read_feather("C:/Users/Graphicsland/Spyder/retention/outputs/customer_summary_last_1_order.feather")
        
    # LikelyChurned
    attributes = [
        'PctHoliday', 'OrderItemPriceTotal', 'TotalDiscount', 'PctWorkHours', 
        'PctSummer', 'PctPresidentialElection', 'PctNormalShipping', 
        ]
    
    
    

    


# Convert object-type columns to category or bool where appropriate
for col in customer_summary.columns:
    if customer_summary[col].dtype == 'object':
        unique_vals = customer_summary[col].dropna().unique()
        if set(unique_vals).issubset({True, False}):
            customer_summary[col] = customer_summary[col].astype(bool)
        else:
            customer_summary[col] = customer_summary[col].astype('category')


# Convert all category columns to numeric codes
for col in customer_summary.select_dtypes(include='category').columns:
    customer_summary[col] = customer_summary[col].cat.codes

# Convert bool to int
for col in customer_summary.select_dtypes(include='bool').columns:
    customer_summary[col] = customer_summary[col].astype(int)



# Make sure all cells are valid
if predict_variable == 'LikelyChurned':
    if 'TotalOrders' not in attributes:
        subset = customer_summary[['CustomerId', 'TotalOrders', predict_variable] + attributes]
    else:
        subset = customer_summary[['CustomerId', predict_variable] + attributes]
    subset = subset[subset['TotalOrders'] >= min_orders]
else:
    subset = customer_summary[['CustomerId', predict_variable] + attributes]
subset = subset.dropna()




X = subset.drop(columns=['CustomerId', predict_variable])

# Choose a label
y = subset[predict_variable]

# y = customer_summary[customer_summary['TotalOrders'] > 1][predict_variable]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train XGBoost
model = XGBClassifier(
    use_label_encoder=False,
    eval_metric='logloss',
    reg_alpha=1,     # L1 regularization (feature selection)
    reg_lambda=1,    # L2 regularization (shrinkage/weight smoothing)
    max_depth=4,     # Lower depth helps reduce overfitting
    subsample=0.8,   # Randomly sample rows for each tree
    colsample_bytree=0.8,  # Randomly sample features per tree
    n_estimators=100,
    learning_rate=0.1
)

model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
print("AUC:", roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]))

## Test the predictive power across several splits (folds) of data
## Rotate Which of the folds is the test set and do it as many times as there are folds
## Stratified KFold maintains the ratio of churned/unchurned in each fold
# cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc')
# print("Stratified AUC scores:", scores)
# print("Mean AUC:", scores.mean())


# Need to put some tooling in to display the number of orders, what's being predicted,
# and the accuracy of the model 
# Feature importance
# Plot top 15 features
plt.figure(figsize=(10, 6))
plot_importance(model, max_num_features=15)
plt.title(f'Top Feature Importances - {predict_variable}')
plt.show()

# SHAP values
# Create explainer
explainer = shap.Explainer(model, X_train)
shap_values = explainer(X_test)
shap.summary_plot(
    shap_values,          # array or Explanation
    X_test,               # the data
    show=False            # <= keep Matplotlib figure open
)

gc.collect()

# 2) grab the figure / axes SHAP just created
fig = plt.gcf()           # current figure
ax  = plt.gca()           # the main axis (beeswarm)

# 3) set the title (or anything else you want)
fig.suptitle(f"SHAP summary for '{predict_variable}'", fontsize=14, y=1.02)

# 4) tidy & display / save
plt.tight_layout()
plt.show()          # or fig.savefig("summary_plot.png", dpi=300)

for name in X_train.columns:
    shap.dependence_plot(name, shap_values.values, X_test, display_features=X_test)


# Force plot for a single prediction
# for i in list(range(123, 133)):
#     shap.plots.waterfall(shap_values[i])
#     model.predict([X_test.iloc[i]])  # where i is the index of 


# # Force a constrained chart
# feature_name = 'AvgDiscount'
# lower, upper = np.percentile(X_test[feature_name], [1, 99])
# inlier_mask = ((X_test[feature_name] >= lower) & (X_test[feature_name] <= upper)).to_numpy()

# # Use filtered SHAP values and features
# shap.dependence_plot(
#     feature_name,
#     shap_values.values[inlier_mask],
#     X_test.iloc[inlier_mask],
#     display_features=X_test.iloc[inlier_mask]
# )
