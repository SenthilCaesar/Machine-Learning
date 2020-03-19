import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.ensemble import RandomForestRegressor
import statistics


features = pd.read_csv('/root/Desktop/analysis/2LeftRightCluster.csv')

# YBOCS is the target variable
labels = np.array(features['YBOCS'])

features= features.drop('YBOCS', axis = 1)

# Saving feature names for later use
feature_list = list(features.columns)

# Convert to numpy array
features = np.array(features)

acc_list = []
Autonomic = []
WM = []
striatum = []
thalamus = []
nac = []

analysis = {'GM': [], 'WM': []}
pred = []

for i in range(1000):

    features, labels = shuffle(features, labels)
    train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.10, random_state = 42)
    rf = RandomForestRegressor(n_estimators = 100, random_state = 42)
    rf.fit(train_features, train_labels)

    # Use the forest's predict method on the test data
    predictions = rf.predict(test_features)
    pred.append(predictions)
    # Calculate the absolute errors
    errors = abs(predictions - test_labels)
    # Print out the mean absolute error (mae)
    # print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')

    # Calculate mean absolute percentage error (MAPE)
    mape = 100 * (errors / test_labels)
    # Calculate and display accuracy
    accuracy = 100 - np.mean(mape)
    acc_list.append(accuracy)
    #print('Accuracy:', round(accuracy, 2), '%.')
    # Get numerical feature importances
    importances = list(rf.feature_importances_)
    # List of tuples with variable and importance
    feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
    # Sort the feature importances by most important first
    feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
    # Print out the feature and importances 
    #[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances];
    
    for pair in feature_importances:
        analysis[pair[0]].append(pair[1])

print(statistics.mean(acc_list))

food_list=list(analysis.values())

print(analysis.keys())
print(statistics.mean(food_list[0]))
print(statistics.mean(food_list[1]))
#print(statistics.mean(food_list[2]))
#print(statistics.mean(food_list[3]))
#print(statistics.mean(food_list[4]))

pred_final = np.zeros(18)
for i in range(1000):
    pred_final += pred[i]
    
pred_final = pred_final / 1000

import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.scatter(test_labels, pred_final)
ax.plot([test_labels.min(), test_labels.max()], [test_labels.min(), test_labels.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Avg Predicted')
plt.show()
