import os
import pickle
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


# Load the 20 Newsgroups dataset
print('loading dataset...')
newsgroups = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(newsgroups.data, newsgroups.target, test_size=0.2, random_state=42)

# Create a tokenizer
print('tokenizing...')
tokenizer = Tokenizer(num_words=5000)  # You can adjust the num_words parameter

# Fit the tokenizer on the training data
tokenizer.fit_on_texts(X_train)

# Convert the text data to sequences
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

# Pad the sequences to have the same length
max_sequence_length = 300  # You can adjust the sequence length
X_train_pad = pad_sequences(X_train_seq, maxlen=max_sequence_length, padding='post')
X_test_pad = pad_sequences(X_test_seq, maxlen=max_sequence_length, padding='post')

# Build the model
model = tf.keras.Sequential([
    Embedding(input_dim=5000, output_dim=128, input_length=max_sequence_length),
    Bidirectional(LSTM(128)),
    Dense(20, activation='softmax')
])

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Print the model summary
model.summary()

# Train the model
model.fit(X_train_pad, y_train, epochs=5, batch_size=64, validation_split=0.2)

# Evaluate the model on the test data
loss, accuracy = model.evaluate(X_test_pad, y_test)
print(f'Accuracy: {accuracy:.2f}')

# Predict the categories for the test data
y_pred = model.predict(X_test_pad)
y_pred = np.argmax(y_pred, axis=-1)

# Print the classification report (including precision, recall, and F1-score)
class_report = classification_report(y_test, y_pred, target_names=newsgroups.target_names)
print('\nClassification Report:\n', class_report)

# Define the directory to save the model
model_dir = 'models/'

# Create the directory if it doesn't exist
os.makedirs(model_dir, exist_ok=True)

# Save the trained model to the model_dir
model.save(os.path.join(model_dir, 'text_classification_model.h5'))

print(f"Model saved to {os.path.join(model_dir, 'text_classification_model.h5')}")

# Save the tokenizer to a separate directory named 'tokenizer'
tokenizer_dir = 'tokenizer/'

# Create the tokenizer directory if it doesn't exist
os.makedirs(tokenizer_dir, exist_ok=True)

# Save the tokenizer to a file in the tokenizer directory
with open(os.path.join(tokenizer_dir, 'tokenizer.pkl'), 'wb') as tokenizer_file:
    pickle.dump(tokenizer, tokenizer_file)

# Save the max_sequence_length to a file in the tokenizer directory
with open(os.path.join(tokenizer_dir, 'max_sequence_length.txt'), 'w') as length_file:
    length_file.write(str(max_sequence_length))

print(f"Model and tokenizer saved to {model_dir} and {tokenizer_dir} respectively.")
