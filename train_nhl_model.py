# The following step is importing all the libraries that will be used to help build the classification model

# Import necessary libraries
import pandas as pd
import numpy as np
import tensorflow as tf  # Used for building and training the neural network model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split  # To split the dataset into training and testing sets
from sklearn.preprocessing import StandardScaler  # For feature scaling (normalization)
from sklearn.metrics import accuracy_score  # To evaluate the model performance

# Loading the dataset from local directory
csv_file = 'NHL 2018-2019 Game Stats.csv'  
dataset = pd.read_csv(csv_file)

# Shows the first few rows of the dataset
print(dataset.head())

print("COLUMNS:\n", dataset.columns)

# Checking for missing values
print(dataset.isnull().sum())

# Drop rows with missing values
dataset = dataset.dropna()

# Selecting the input features (x) and the label (y)
# Create 'won' column: 1 if GF > GA, else 0
dataset['won'] = dataset['GF'] > dataset['GA']
dataset['won'] = dataset['won'].astype(int)

# Selecting input features
x = dataset[['GF', 'SF', 'FF', 'CF', 'FF%', 'SF%']]
y = dataset['won']

# Splitting data into training and testing sets 80% training and 20% testing
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Feature scaling done to Normalize the feature values to improve model performance
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)  # Fit scaler to training data and transform it
x_test = scaler.transform(x_test)  

# Building the neural network model for binary classification
model = Sequential()

# Input and first hidden layer with 10 neurons and ReLU activation added
model.add(Dense(10, input_dim=x_train.shape[1], activation='relu'))

# Adding the second hidden layer with 5 neurons
model.add(Dense(5, activation='relu'))

# Adding the output layer for binary classification (sigmoid activation for binary)
model.add(Dense(1, activation='sigmoid'))


model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Training the model on the training data
model.fit(x_train, y_train, epochs=40, batch_size=32)

# Evaluating the model's performance 
y_pred = model.predict(x_test)
y_pred = (y_pred > 0.5)  

# Calculating and printing the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print(f"The model's accuracy on the test set is: {accuracy * 100:.2f}%")


model.save("nhl_model.h5")
print(" Model saved successfully as nhl_model.h5")
