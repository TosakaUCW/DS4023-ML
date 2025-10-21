import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import train_test_split

# Load the 2.2-Exercise.csv dataset
data = pd.read_csv('../docs/2.2-Exercise.csv')

# Use High column as features (X) and Target column as prediction target (y)
x = data['High'].values.reshape(-1, 1)  # Reshape to 2D array for sklearn
y = data['Target'].values

# Split the data set into testing and training data
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0)

# Create a linear regression object
regression = linear_model.LinearRegression()

# Train the model using the training set
regression.fit(x_train, y_train)

# Make predictions using the testing set
y_predictions = regression.predict(x_test)

# Plot the data
sns.set_style("darkgrid")
sns.regplot(x=x_test.flatten(), y=y_test, fit_reg=False)
plt.plot(x_test, y_predictions, color='black')

# Add labels and title
plt.xlabel('High')
plt.ylabel('Target')
plt.title('Linear Regression: High vs Target')

plt.tight_layout()
plt.show()
