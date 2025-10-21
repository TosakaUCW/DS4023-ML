import numpy as np
import pandas as pd
from sklearn.naive_bayes import BernoulliNB
from sklearn.preprocessing import LabelEncoder

# Load the basketball.csv dataset
data = pd.read_csv('../docs/basketball.csv')

# The features in X are the first 4 columns: outlook, temp, humidity, windy
# The target y is the last column: play

# Separate features and target
X = data.iloc[:, :-1]  # First 4 columns as features
y = data.iloc[:, -1]   # Last column as target

# Convert categorical features to numerical values using LabelEncoder
label_encoders = {}
X_encoded = X.copy()

for column in X.columns:
    le = LabelEncoder()
    X_encoded[column] = le.fit_transform(X[column])
    label_encoders[column] = le

# Convert to numpy array for sklearn
X_array = X_encoded.values

# This is the code we need for the Bernoulli model
clf = BernoulliNB()
# We train the model on our data
clf.fit(X_array, y)

# Now we can make a prediction on what class new data belongs to
print("Our data set represents basketball games and weather conditions.\n")
print("We have trained a Bernoulli model on our data set.\n")
print("Features: outlook, temp, humidity, windy")
print("Target: play (yes/no)\n")

# Let's make a prediction for a new weather condition
# Example: sunny, hot, high, false
print("Let's consider a new input that:")
print("   outlook: sunny")
print("   temp: hot")
print("   humidity: high")
print("   windy: false\n")

# Encode the new input using the same encoders
new_input = ['sunny', 'hot', 'high', 'false']
new_input_encoded = []
for i, value in enumerate(new_input):
    encoded_value = label_encoders[X.columns[i]].transform([value])[0]
    new_input_encoded.append(encoded_value)

print("What does our model think this should be?")
print("Answer: %s!" % clf.predict([new_input_encoded])[0])
