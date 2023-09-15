import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.datasets import fetch_20newsgroups


def load(model_path, tokenizer_path, max_sequence_path):
    """
    Load the pre-trained model weights.
    Returns:
        model: A TensorFlow/Keras model.
        tokenizer: A Tokenizer for text preprocessing.
        max_sequence_length: The maximum sequence length used during training.
    """
    # Load the model
    model = tf.keras.models.load_model(model_path)
    
    # Load the tokenizer used for preprocessing
    with open(tokenizer_path, 'rb') as tokenizer_file:
        tokenizer = pickle.load(tokenizer_file)
    
    # Load the max_sequence_length used during training
    with open(max_sequence_path, 'r') as length_file:
        max_sequence_length = int(length_file.read())
    
    return model, tokenizer, max_sequence_length

def predict(text, model, tokenizer, max_sequence_length):
    """
    Make a prediction using the loaded model.
    Args:
        text (str): The input text for prediction.
        model: The pre-trained TensorFlow/Keras model.
        tokenizer: A Tokenizer for text preprocessing.
        max_sequence_length: The maximum sequence length used during training.
    Returns:
        prediction (str): The predicted class as a string.
    """
    # Tokenize and pad the input text
    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=max_sequence_length, padding='post')
    
    # Make a prediction using the model
    predicted_class = model.predict(padded_sequence)[0]
    predicted_class = np.argmax(predicted_class, axis=-1)
    
    # Convert the predicted class to its corresponding label
    label_mapping = fetch_20newsgroups()['target_names']
    predicted_label = label_mapping[predicted_class]
    
    return predicted_label


if __name__ == '__main__':

    # Define the directory where the model is saved
    model_path = 'models/text_classification_model.h5'
    tokenizer_path = 'tokenizer/tokenizer.pkl'
    max_sequence_path = 'tokenizer/max_sequence_length.txt'

    model, tokenizer, max_len = load(model_path, tokenizer_path, max_sequence_path)

    query = 'The space shuttle launch has been delayed due to technical issues.'

    print(predict(query, model, tokenizer, max_len))

