import pandas as pd
from xgboost import XGBClassifier, plot_importance
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, roc_auc_score
import matplotlib.pyplot as plt
import shap

predict_variable = 'LikelyChurned'
# predict_variable = 'OneAndDone'


# Load your data
customer_summary = pd.read_feather("C:/Users/Graphicsland/Spyder/retention/outputs/customer_summary.feather")

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

# LikelyChurned
# Drop obvious identifier / label leakage
X = customer_summary[['TotalOrders', 'AvgOrderItemTotal', 'AvgPriceTotal',
       'AvgDiscount', 'AvgQuantity', 'ModeHour', 'ModeDayOfWeek',
       'QuarterlyPattern',
       'MonthlyPattern', 'SeasonalPattern', 'ElectionPattern',
       'PctMorning', 'PctAfternoon', 'PctEvening', 'PctEarlyMorning',
       'PctWorkHours', 'PctWorkday', 'PctHoliday', 'PctSummer',
       'PctElectionSeason', 'PctPresidentialElection', 'PctFirstWeek',
       'PctLastWeek', 'PctRushed', 'PctExpedited', 'PctNormal', 'PctNo-Ship'
]]

# OneAndDone
# X = X[X['TotalOrders'] > 1]

# X = customer_summary[[
#     'DaysSinceLastOrder', 'PctHoliday','AvgQuantity', 'PctWorkHours', 
#     'PctSummer', 'PctPresidentialElection', 'PctExpedited', 
# ]]


# X = customer_summary[[
#     'PctHoliday','AvgQuantity', 'PctWorkHours', 
#     'PctSummer', 'PctPresidentialElection', 'PctExpedited', 
# ]]



# Choose a label
y = customer_summary[predict_variable]
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



# Feature importance
# Plot top 15 features
plt.figure(figsize=(10, 6))
plot_importance(model, max_num_features=15)
plt.title('Top 15 Feature Importances')
plt.show()

# SHAP values
# Create explainer
explainer = shap.Explainer(model, X_train)
shap_values = explainer(X_test)
shap.summary_plot(shap_values, X_test)

# Force plot for a single prediction
i = 1000
shap.plots.waterfall(shap_values[i])
model.predict([X_test.iloc[i]])  # where i is the index of 
