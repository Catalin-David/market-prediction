import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Load your data
# For this example, replace this with the path to your CSV file
data = pd.read_csv('stock_market_data.csv')

# Assuming the CSV has 'Date' and 'Close' columns
data['Date'] = pd.to_datetime(data['Date'])
data.sort_values('Date', inplace=True)

data.reset_index(inplace=True)
data['Days'] = (data['Date'] - data['Date'].min()).dt.days

# Use 'Days' as the feature for the regression
X = data['Days'].values.reshape(-1, 1)  # Features (number of days)
y = data['Close'].values         # Target variable (closing price)

# Split the data into training/testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Create linear regression object
model = LinearRegression()

# Train the model using the training sets
model.fit(X_train, y_train)

# Make predictions using the testing set
y_pred = model.predict(X_test)

# The coefficients
print('Coefficients: \n', model.coef_)

# Plot outputs (optional)
plt.scatter(X_test, y_test,  color='black')
plt.plot(X_test, y_pred, color='blue', linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()
