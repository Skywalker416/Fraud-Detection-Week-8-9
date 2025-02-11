import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, LSTM, GRU, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import os

# Define data directory
data_dir = "./data/"  # Update this path if needed

# Load credit card dataset
credit_data = pd.read_csv(os.path.join(data_dir, "creditcard.csv"))

# Feature and Target Separation
X_credit = credit_data.drop(columns=['Class'])
y_credit = credit_data['Class']

# Train-Test Split
X_credit_train, X_credit_test, y_credit_train, y_credit_test = train_test_split(X_credit, y_credit, test_size=0.2, random_state=42)

# Standardize Features
scaler = StandardScaler()
X_credit_train = scaler.fit_transform(X_credit_train)
X_credit_test = scaler.transform(X_credit_test)

# Reshape for CNN/LSTM input
X_credit_train = X_credit_train.reshape(X_credit_train.shape[0], X_credit_train.shape[1], 1)
X_credit_test = X_credit_test.reshape(X_credit_test.shape[0], X_credit_test.shape[1], 1)

# CNN Model
cnn_model = Sequential([
    Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(X_credit_train.shape[1], 1)),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])
cnn_model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Train CNN
cnn_model.fit(X_credit_train, y_credit_train, epochs=10, batch_size=32, validation_data=(X_credit_test, y_credit_test))

# LSTM Model
lstm_model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(X_credit_train.shape[1], 1)),
    LSTM(32),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])
lstm_model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Train LSTM
lstm_model.fit(X_credit_train, y_credit_train, epochs=10, batch_size=32, validation_data=(X_credit_test, y_credit_test))

# RNN Model
rnn_model = Sequential([
    GRU(64, return_sequences=True, input_shape=(X_credit_train.shape[1], 1)),
    GRU(32),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])
rnn_model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Train RNN
rnn_model.fit(X_credit_train, y_credit_train, epochs=10, batch_size=32, validation_data=(X_credit_test, y_credit_test))
