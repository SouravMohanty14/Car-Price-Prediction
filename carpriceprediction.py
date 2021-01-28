import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('car data.csv')
pd.set_option('display.width', 500)
pd.set_option('display.max_columns', len(df.columns))
print(df.head())
print()
print(df.shape)

# Print Unique Values of Categorical Features
print(df['Fuel_Type'].unique())
print(df['Seller_Type'].unique())
print(df['Transmission'].unique())
print(df['Owner'].unique())

# Check Missing or Null Values
print(df.isnull().sum())

print(df.describe())

# Skip Car Name since it will not contribute anything; Cannot Judge based on Car name
print(df.columns)

final_dataset = df[
    ['Year', 'Selling_Price', 'Present_Price', 'Kms_Driven', 'Fuel_Type', 'Seller_Type', 'Transmission', 'Owner']]

# Here we are going to Convert the Year Column to No of Years
final_dataset['Current_Year'] = 2021
final_dataset['No_of_Years'] = final_dataset['Current_Year'] - final_dataset['Year']

final_dataset.drop(['Year', 'Current_Year'], axis=1, inplace=True)
# inplace for a permanent change
print(final_dataset.head())

# Categorical Features
final_dataset = pd.get_dummies(final_dataset, drop_first=True)
# First Col Deleted to avoid Dummy Variable Trap
# print(final_dataset.head())
print(final_dataset.columns)

# Check Correlation
# print(final_dataset.corr())
# # Use Pairplot to visualize correlation of each & every features
# sns.pairplot(final_dataset)
# # plt.show()
# corrmat = final_dataset.corr()
# top_corr_features = corrmat.index
# plt.figure(figsize=(20, 20))
# # plot heat map
# g = sns.heatmap(final_dataset[top_corr_features].corr(), annot=True, cmap="RdYlGn")
# plt.show()

# Independent & dependent Features
X = final_dataset.iloc[:,1:]
y = final_dataset.iloc[:,0] # Selling Price as y

print()
print(X.head())
print(y.head())

# Feature Importance
from sklearn.ensemble import ExtraTreesRegressor
model = ExtraTreesRegressor()
model.fit(X,y)

# use inbuilt class feature_importances of tree based classifiers
print(model.feature_importances_) #get scores of all independent columns

# Plot graph of feature importance scores for better visualization
# feat_importances = pd.Series(model.feature_importances_, index=X.columns) #create a series
# feat_importances.nlargest(5).plot(kind='barh') #plot a bar graph of N best features
# plt.show()

# Train Test Split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
print(X_train.shape)

# Creating Model
from sklearn.ensemble import RandomForestRegressor
rf_random = RandomForestRegressor()