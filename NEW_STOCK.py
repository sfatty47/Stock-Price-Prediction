#!/usr/bin/env python
# coding: utf-8

# **Stock Market Prediction**

# The goal of this project is use supervised learning techniques to predict the future price of a given stock based on its past price data.

# Importing Packages 

# In[1]:


import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor


# In[2]:


import seaborn as sns


# In[3]:


data = pd.read_csv("C:/Users/Sankung/Downloads/sp500_daily_ratios_20yrs.csv")


# In[4]:


data.head()


# In[5]:


data.tail()


# In[6]:


len(data)


# In[7]:


data.columns


# In[8]:


data.isnull().sum()


# In[9]:


data.dtypes


# Data Cleaning 

# In[17]:


corr = data.corr()

plt.figure(figsize=(10, 8))  # Adjust the figure size as per your preference

sns.heatmap(corr, annot=True, fmt=".2f", linewidths=0.5, cmap='plasma', annot_kws={"fontsize": 8})

plt.title('Correlation Matrix')
plt.show()


# In[10]:


# Drop unnecessary columns
data = data.drop(['Open'], axis=1)

# Find highly correlated variables
corr_matrix = data.corr().abs()

# Select upper triangle of correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

# Find index of feature columns with correlation greater than 0.75
to_drop = [column for column in upper.columns if any(upper[column] > 0.75)]

# Drop highly correlated columns
data = data.drop(to_drop, axis=1)


# In[19]:


corr = data.corr()

plt.figure(figsize=(10, 8))  # Adjust the figure size as per your preference

sns.heatmap(corr, annot=True, fmt=".2f", linewidths=0.5, cmap='plasma', annot_kws={"fontsize": 8})

plt.title('Correlation Matrix')
plt.show()


# In[11]:


# drop the column
data = data.drop(columns=['ROA - Return On Assets', 'ROE - Return On Equity', 'Return On Tangible Equity'])


# In[ ]:


from sklearn.impute import KNNImputer
# create KNNImputer object
imputer = KNNImputer(n_neighbors=2)

# impute missing values in column 'C'
data['ROE - Return On Equity'] = imputer.fit_transform(data[['ROE - Return On Equity']])
data['Return On Tangible Equity'] = imputer.fit_transform(data[['Return On Tangible Equity']])


# Data Analysis 

# In[ ]:


import seaborn as sns

sns.pairplot(data)
plt.tight_layout()


# In[37]:


amazon_data['Date'] = pd.to_datetime(amazon_data['Date'])  

plt.figure(figsize=(16, 8))

colors = sns.color_palette('Blues')

ax = plt.gca()
ax.set_facecolor('#f7f7f7')
ax.patch.set_alpha(0.8)

plt.title('Amazon Close Price History', fontsize=20)
plt.plot(amazon_data['Date'], amazon_data['Close'], color=colors[4], linewidth=2.5)
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize=18)

plt.xticks(rotation=45, ha='right', fontsize=12)
plt.yticks(fontsize=12)

plt.grid(True, linestyle='--', alpha=0.5)

ax.xaxis.grid(color='white', linestyle='-', linewidth=0.5)
ax.yaxis.grid(color='white', linestyle='-', linewidth=0.5)

sns.despine()

plt.gca().set_rasterization_zorder(1)

plt.tight_layout()

plt.show()


# In[47]:


plt.figure(figsize=(16, 8))
plt.title('Amazon Volume History', fontsize=20)
plt.plot(amazon_data['Volume'], color='b', linewidth=2)
plt.xlabel('Date', fontsize=18)
plt.ylabel('Volume of the Stock ($)', fontsize=18)

plt.xticks(rotation=45, ha='right', fontsize=12)
plt.yticks(fontsize=12)

plt.grid(True, linestyle='--', alpha=0.5)

# Customize background color
plt.gca().set_facecolor('#f7f7f7')
plt.gcf().set_facecolor('#ffffff')

plt.show()


# There are 500 large publicly trading companies in United States in this dataset so this graph is for Amazon only.

# In[12]:


unique_years = len(data['year'].unique())
unique_years


# Regression Models 

# In[13]:


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

X = data.drop(['Ticker', 'Date', 'Close'], axis=1)
y = data['Close']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[14]:


num_features = X_train.shape[1]
print("Number of features:", num_features)


# In[15]:


from sklearn.metrics import r2_score, mean_squared_error

# Train the model
reg = LinearRegression()
reg.fit(X_train, y_train)

# Make predictions on the training and test sets
y_train_pred = reg.predict(X_train)
y_test_pred = reg.predict(X_test)

# Calculate R-squared and RMSE for the training and test sets
reg_train_r2 = r2_score(y_train, y_train_pred)
reg_test_r2 = r2_score(y_test, y_test_pred)
reg_train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
reg_test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

# Print the R-squared and RMSE values for the linear regression model
print(f"Linear regression training set R-squared: {reg_train_r2:.3f}")
print(f"Linear regression test set R-squared: {reg_test_r2:.3f}")
print(f"Linear regression training set RMSE: {reg_train_rmse:.3f}")
print(f"Linear regression test set RMSE: {reg_test_rmse:.3f}")


# In[16]:


from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np

# Create a Ridge object with regularization parameter alpha
ridge = Ridge(alpha=0.1)

# Fit the Ridge model to the training data
ridge.fit(X_train, y_train)

# Make predictions using the Ridge model on the test data
y_pred_test = ridge.predict(X_test)

# Calculate R2 for the test predictions
r2_test = r2_score(y_test, y_pred_test)
print("Test R2:", r2_test)

# Calculate RMSE for the test predictions
rmse_ridge_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
print("Test RMSE:", rmse_ridge_test)

# Make predictions using the Ridge model on the training data
y_pred_train = ridge.predict(X_train)

# Calculate R2 for the training predictions
r2_train = r2_score(y_train, y_pred_train)
print("Training R2:", r2_train)

# Calculate RMSE for the training predictions
rmse_ridge_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
print("Training RMSE:", rmse_ridge_train)


# It looks like the model is not performing very well, since the R-squared values are quite low (around 0.08) and the RMSE values are quite high (around 92). This suggests that the model is not capturing the variability in the data very well, and is making predictions that are not very accurate.

# In[60]:


from sklearn.ensemble import RandomForestRegressor
# Train the random forest model
rfr = RandomForestRegressor(n_estimators=12, max_depth=3, min_samples_leaf=3, max_features=2, random_state=42)
rfr.fit(X_train, y_train)

# Make predictions on the training and test sets
y_train_pred = rfr.predict(X_train)
y_test_pred = rfr.predict(X_test)

# Calculate R-squared and RMSE for the training and test sets
rfr_train_r2 = r2_score(y_train, y_train_pred)
rfr_test_r2 = r2_score(y_test, y_test_pred)
rfr_train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
rfr_test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

# Print the R-squared and RMSE values for the random forest model
print(f"Random forest training set R-squared: {rfr_train_r2:.3f}")
print(f"Random forest test set R-squared: {rfr_test_r2:.3f}")
print(f"Random forest training set RMSE: {rfr_train_rmse:.3f}")
print(f"Random forest test set RMSE: {rfr_test_rmse:.3f}")


# In[61]:


import matplotlib.pyplot as plt

# Calculate the residuals
train_residuals = y_train - y_train_pred
test_residuals = y_test - y_test_pred

# Create two separate residual plots for training and test sets
plt.figure(figsize=(16, 6))

# Residual plot for the training set
plt.subplot(1, 2, 1)
plt.scatter(y_train_pred, train_residuals, color='blue', label='Training set')
plt.axhline(y=0, color='black', linestyle='--')
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.title('Residual Plot - Training Set')
plt.legend()

# Residual plot for the test set
plt.subplot(1, 2, 2)
plt.scatter(y_test_pred, test_residuals, color='red', label='Test set')
plt.axhline(y=0, color='black', linestyle='--')
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.title('Residual Plot - Test Set')
plt.legend()

plt.tight_layout()
plt.show()


# The random forest model's performance can be summarized based on the R-squared and root mean squared error (RMSE) metrics. The R-squared value measures the proportion of variance in the target variable explained by the model. For the training set, the random forest achieved an R-squared of 0.193, indicating that the model can explain approximately 19.3% of the variance in the target variable. Similarly, for the test set, the R-squared value was 0.196, suggesting that the model's performance generalizes relatively well to unseen data. The RMSE metric quantifies the average difference between the predicted values and the actual values. In this case, the random forest achieved an RMSE of 110.929 for the training set and 111.290 for the test set. These values indicate the average prediction error, with lower values indicating better predictive accuracy. Overall, the random forest model performed reasonably well, explaining a moderate amount of variance in the target variable and achieving relatively low prediction errors.

# In[62]:


from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Create a Lasso object with regularization parameter alpha
lasso = Lasso(alpha=0.1)

# Fit the Lasso model to the training data
lasso.fit(X_train, y_train)

# Make predictions using the Lasso model on the test data
y_pred_test = lasso.predict(X_test)

# Calculate RMSE for the test predictions
rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
print("Test RMSE:", rmse_test)

# Calculate R2 for the test predictions
r2_test = r2_score(y_test, y_pred_test)
print("Test R2:", r2_test)

# Make predictions using the Lasso model on the training data
y_pred_train = lasso.predict(X_train)

# Calculate RMSE for the training predictions
rmse_train_lasso = np.sqrt(mean_squared_error(y_train, y_pred_train))
print("Training RMSE:", rmse_train_lasso)

# Calculate R2 for the training predictions
r2_train = r2_score(y_train, y_pred_train)
print("Training R2:", r2_train)


# Test RMSE of 118.5149698298507 suggests that the predictions are off by approximately $92 on average.
# 
# The R-squared (R2) score is a statistical measure that represents how close the data fits to the regression line. The R2 score ranges from 0 to 1, where 1 indicates a perfect fit. In this case, the Test R2 score of 0.0830 indicates that only about 8.3% of the variance in the stock price is explained by the model.
# 
# Similarly, the Training RMSE of 92.7626 indicates that the training set predictions are off by approximately $118 on average, while the Training R2 of 0.0847 indicates that only about 8.5% of the variance in the target variable is explained by the model on the training set.

# In[63]:


import matplotlib.pyplot as plt
import seaborn as sns

# plot the actual vs predicted values for training data
plt.scatter(y_train, y_train_pred, c='blue', alpha=0.5)
sns.regplot(x=y_train, y=y_train_pred, scatter=False, color='red')
plt.xlabel('Actual values')
plt.ylabel('Predicted values')
plt.title('Training data')
plt.show()

# plot the actual vs predicted values for test data
plt.scatter(y_test, y_test_pred, c='red', alpha=0.5)
sns.regplot(x=y_test, y=y_test_pred, scatter=False, color='blue')
plt.xlabel('Actual values')
plt.ylabel('Predicted values')
plt.title('Test data')
plt.show()


# In[64]:


import matplotlib.pyplot as plt

# Create a bar graph of the RMSE values
rmse_values = [rmse_ridge_test, rfr_test_rmse, reg_test_rmse, rmse_train_lasso]
models = ['Ridge', 'Random Forest', 'Linear Regression', 'Lasso']

plt.bar(models, rmse_values)
plt.title('RMSE for Stock Price Prediction Models')
plt.xlabel('Model')
plt.ylabel('RMSE')

# Rotate the x-axis labels by 45 degrees
plt.xticks(rotation=45)

plt.show()


# In[65]:


from sklearn.preprocessing import MinMaxScaler

# Create a scaler object
scaler = MinMaxScaler(feature_range=(0, 1))

# Scale the outcome values
y_train_scaled = scaler.fit_transform(y_train.to_numpy().reshape(-1, 1)).flatten()
y_test_scaled = scaler.transform(y_test.to_numpy().reshape(-1, 1)).flatten()
y_train_pred_scaled = scaler.transform(y_train_pred.reshape(-1, 1)).flatten()
y_test_pred_scaled = scaler.transform(y_test_pred.reshape(-1, 1)).flatten()

# Plot the scaled outcome values
plt.figure(figsize=(12, 6))
plt.plot(y_train_scaled, label='Actual Training Labels')
plt.plot(y_train_pred_scaled, label='Training Predictions')
plt.plot(range(len(y_train_scaled), len(y_train_scaled) + len(y_test_scaled)), y_test_scaled, label='Actual Testing Labels')
plt.plot(range(len(y_train_scaled), len(y_train_scaled) + len(y_test_scaled)), y_test_pred_scaled, label='Testing Predictions')
plt.xlabel('Index')
plt.ylabel('Scaled Value')
plt.legend()
plt.show()


# In[66]:


len(y_train)


# In[67]:


# Create a figure
fig, ax = plt.subplots(figsize=(8, 6))

# Create a boxplot for the training data
train_data = [y_train, y_train_pred]
bp = ax.boxplot(train_data, labels=['Actual', 'Predicted'], showfliers=False, patch_artist=True, notch=True)

# Set colors for the boxplots
colors = ['lightblue', 'lightgreen']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)

# Add a title and labels
ax.set_title('Boxplot of Actual and Predicted Training Labels', fontsize=16)
ax.set_ylabel('Stock Price', fontsize=14)

# Set tick label font size
ax.tick_params(axis='both', which='major', labelsize=12)

# Show the plot
plt.show()

# Create another figure for the testing data
fig, ax = plt.subplots(figsize=(8, 6))

# Create a boxplot for the testing data
test_data = [y_test, y_test_pred]
bp = ax.boxplot(test_data, labels=['Actual', 'Predicted'], showfliers=False, patch_artist=True, notch=True)

# Set colors for the boxplots
colors = ['lightblue', 'lightgreen']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)

# Add a title and labels
ax.set_title('Boxplot of Actual and Predicted Testing Labels', fontsize=16)
ax.set_ylabel('Stock Price', fontsize=14)

# Set tick label font size
ax.tick_params(axis='both', which='major', labelsize=12)

# Show the plot
plt.show()


# In[ ]:


np.max(y_test)


# In[ ]:


np.max(y_test_pred)


# In[ ]:


# DECISION


# In[31]:


for i in range(min(len(y_test), 10)):
    if y_test_pred[i] > y_test.iloc[i]:
        print("Buy stock!")
    elif y_test_pred[i] < y_test.iloc[i]:
        print("Sell stock!")
    else:
        print("Do nothing.")


# In[27]:


# Decision making of whether to buy or sell the stock.


# ## CLASSIFICATION

# In[17]:


data['Price_Diff'] = data['Close'].diff().fillna(0)
data['StockP'] = (data['Price_Diff'] > 0).astype(int)


# In[18]:



X = data.drop(['Ticker', 'Date', 'Close', 'StockP','Price_Diff'], axis=1)
y = data['StockP']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[19]:


# Import required libraries

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Create a logistic regression model
lr_model = LogisticRegression()

# Train the model on the training data
lr_model.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = lr_model.predict(X_test)

# Evaluate the model using classification report
print(classification_report(y_test, y_pred))


# The logistic classifier's performance, as summarized by the classification report, reveals mixed results. The model achieved high precision for class 1, indicating that when it predicted an instance as class 1, it was correct around 52% of the time. However, the precision for class 0 was 0.00, meaning that none of the instances predicted as class 0 were correct. The recall for class 1 was 1.00, indicating that the model captured all instances of class 1. However, the recall for class 0 was 0.00, implying that the model failed to identify any instances of class 0. Consequently, the F1-score for class 1 was 0.68, representing a balance between precision and recall, while the F1-score for class 0 was 0.00. The overall accuracy of the model was 0.52, suggesting that it performed moderately in terms of correct predictions. Nonetheless, the imbalanced precision, recall, and F1-scores indicate that the model's performance might be heavily skewed towards class 1, requiring further evaluation and potentially more balanced training data.
# 

# In[ ]:


print('len(y_test) =', len(y_test))


# In[ ]:


from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Standardize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Fit the PCA model
pca = PCA(n_components=2) # specify number of components to retain
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# Check the explained variance ratio
print(pca.explained_variance_ratio_)


# In[ ]:


from sklearn.linear_model import LogisticRegression

# Create a logistic regression model
logreg = LogisticRegression()

# Fit the model using the transformed data
logreg.fit(X_train_pca, y_train)

# Predict the target variable for the test set
y_pred_lr = logreg.predict(X_test_pca)

# Calculate the accuracy of the model
accuracy = logreg.score(X_test_pca, y_test)

# Print the accuracy of the model
print("Accuracy:", accuracy)


# # Random Forest

# In[ ]:


from sklearn.ensemble import RandomForestClassifier

# Initialize the classifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the classifier
rf.fit(X_train, y_train)

# Predict the labels of the test set
y_pred = rf.predict(X_test)

# Evaluate the accuracy of the classifier
accuracy = rf.score(X_test, y_test)
print(f"Accuracy: {accuracy}")


# In[ ]:


# Classification report
cr = classification_report(y_test, y_pred)
print('Classification Report:')
print(cr)


# The random forest model's performance, as summarized by the classification report, suggests that it achieved moderate accuracy in classifying instances into two classes. With an accuracy of 0.50, the model's predictions were correct approximately half of the time. The precision and recall scores for both classes (0 and 1) were fairly similar, with values of 0.49 and 0.52, respectively. This indicates that the model had similar success rates in correctly identifying instances for both classes. The F1-scores, which consider both precision and recall, were also around 0.50 for both classes. Overall, the model's performance appears to be modest, as reflected by the balanced precision, recall, and F1-score values, suggesting that it may require further improvements to achieve more accurate and reliable predictions

# In[ ]:


from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print('Confusion Matrix:')
print(cm)


# In[ ]:


# Print confusion matrix
print("Confusion Matrix:")
print("--------------------")
print("| TN | FP |")
print("--------------------")
print(f"| {cm[0, 0]}  | {cm[0, 1]}  |")
print("--------------------")
print("| FN | TP |")
print("--------------------")
print(f"| {cm[1, 0]}  | {cm[1, 1]}  |")
print("--------------------")


# In[ ]:


# ROC curve
y_pred_prob = rf.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.show()


# A ROC curve with a 45 degrees line with a positive slope indicates that the True Positive Rate (TPR) and False Positive Rate (FPR) are the same, meaning that the classifier is no better than random guessing. Ideally, we want a ROC curve that lies above this line and is closer to the top left corner, where TPR is high and FPR is low. This would indicate a better classifier with higher accuracy.

# In[ ]:


from keras.models import Sequential
from keras.layers import LSTM, Dense

# define the model architecturew
model = Sequential()
model.add(LSTM(64, input_shape=(timesteps, input_dim)))
model.add(Dense(1, activation='sigmoid'))

# compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# train the model
model.fit(x_train, y_train, batch_size=32, epochs=10, validation_data=(x_val, y_val))

# # evaluate the model
score = model.evaluate(x_test, y_test, batch_size=32)

# # make predictions
y_pred = model.predict(x_new_data)


# In[ ]:


print(y_pred)


# In[ ]:


data.columns


# In[ ]:


close_df = data[['Date', 'Close']]
close_df.set_index('Date', inplace=True)
close_df.plot(figsize=(10,6))

plt.title('S&P 500 Closing Prices (1998-2018)')
plt.xlabel('Year')
plt.ylabel('Price')
plt.show()


# In[ ]:


sns.set_style('whitegrid')
plt.figure(figsize=(10,6))
sns.scatterplot(x='Close', y='Volume', data=data, alpha=0.5, color='teal')
plt.title('Volume vs. Closing Price')
plt.xlabel('Closing Price')
plt.ylabel('Volume')
plt.show()


# In[ ]:


# Create the distribution plot
sns.set_style('dark')
plt.figure(figsize=(10, 6))
sns.distplot(data['Close'], color='purple')
plt.title('Distribution of Closing Prices')
plt.xlabel('Closing Price')
plt.show()


# In[ ]:


ratios_df = data[['Date', 'Asset Turnover', 'Current Ratio', 'Debt/Equity Ratio', 'Net Profit Margin']]
ratios_df.set_index('Date', inplace=True)

# Create the line plot
sns.set_style('darkgrid')
plt.figure(figsize=(10, 6))
sns.lineplot(data=ratios_df, palette='Set3')
plt.title('Financial Ratios over Time')
plt.xlabel('Year')
plt.show()


# In[ ]:


quarterly_ratios = data.groupby('quarter')[['Asset Turnover', 'Current Ratio', 'Debt/Equity Ratio', 'Net Profit Margin']].mean()

# Create the bar chart
sns.set_style('whitegrid')
plt.figure(figsize=(10, 6))
sns.barplot(x=quarterly_ratios.index, y='Net Profit Margin', data=quarterly_ratios, palette='pastel')
plt.title('Average Net Profit Margin by Quarter')
plt.xlabel('Quarter')
plt.ylabel('Net Profit Margin')
plt.show()


# In[ ]:


import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


# In[ ]:


from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense


# In[ ]:



# set the number of time steps and features
timesteps = 10
features = data.shape[1] - 2

# scale the data
scaler = MinMaxScaler(feature_range=(0, 1))
data[data.columns[2:]] = scaler.fit_transform(data[data.columns[2:]])

# split the data into smaller chunks
data_chunks = []
for i in range(0, len(data)-timesteps, 100):
    data_chunks.append(df.iloc[i:i+timesteps, :])

# create training and test sets
split = int(len(data_chunks)*0.8)
X_train = np.array([chunk.iloc[:, 2:].values for chunk in data_chunks[:split]])
y_train = np.array([chunk.iloc[-1, 3] for chunk in data_chunks[:split]])
X_test = np.array([chunk.iloc[:, 2:].values for chunk in data_chunks[split:]])
y_test = np.array([chunk.iloc[-1, 3] for chunk in data_chunks[split:]])

# define the LSTM model
model = Sequential()
model.add(LSTM(128, input_shape=(timesteps, features)))
model.add(Dense(1, activation='linear'))

# compile the model
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# evaluate the model
loss, mae = model.evaluate(X_test, y_test, batch_size=32)
print('Test Loss:', loss)
print('Test MAE:', mae)

# make predictions on new data
X_new = df.iloc[-timesteps:, 2:].values
X_new = np.reshape(X_new, (1, timesteps, features))
y_pred = model.predict(X_new)


# In[25]:


import xgboost as xgb
from sklearn.metrics import accuracy_score

# Create the XGBoost classifier
model = xgb.XGBClassifier()

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


# An accuracy of 0.5242503329910705 indicates that the model's predictions were correct approximately 52.4% of the time. While this accuracy score might suggest some predictive power, it is relatively low, implying that the model's performance is not very reliable or accurate.

# In[26]:



# Get feature importance
importance = model.feature_importances_
feature_names = X_train.columns

# Sort feature importance in descending order
sorted_indices = importance.argsort()[::-1]
sorted_importance = importance[sorted_indices]
sorted_feature_names = feature_names[sorted_indices]

# Plot feature importance
plt.figure(figsize=(10, 6))
plt.barh(range(len(sorted_importance)), sorted_importance, align='center', color='skyblue')
plt.yticks(range(len(sorted_importance)), sorted_feature_names)
plt.xlabel('Importance')
plt.ylabel('Features')
plt.title('Feature Importance')
plt.tight_layout()
plt.show()


# We can observe that the stock market exhibits discernible trends and seasonality patterns. In our analysis, we found that the 'Year' and 'Quarter' variables emerged as the most significant features, highlighting their importance in understanding the dynamics of the stock market.
