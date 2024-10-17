import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

VOCAB_SIZE = 40000
MAX_LEN = 100
MODEL_PATH = "model10.h5"
TOKENIZER_PATH = "tokenizer1.pickle"

# Load the saved model
model = load_model(MODEL_PATH)
model.summary()
with open(TOKENIZER_PATH, 'rb') as handle:
    tokenizer = pickle.load(handle)


def encode_texts(text_list):
    encoded_texts = []
    for text in text_list:
        tokens = tf.keras.preprocessing.text.text_to_word_sequence(text)
        tokens = [tokenizer.word_index[word] if word in tokenizer.word_index else 0 for word in tokens]
        encoded_texts.append(tokens)
    return pad_sequences(encoded_texts, maxlen=MAX_LEN, padding='post', value=VOCAB_SIZE-1)

def encode_text(text):
    
    tokens = tf.keras.preprocessing.text.text_to_word_sequence(text)
    tokens = [
        tokenizer.word_index[word] if word in tokenizer.word_index and tokenizer.word_index[word] < VOCAB_SIZE else 0
        for word in tokens
    ]
    return pad_sequences([tokens], maxlen=MAX_LEN, padding='post', value=VOCAB_SIZE-1)

def predict_sentiments(text_list):
    print(text_list)
    encoded_inputs = encode_text(text_list)
    pred =np.array(model.predict(encoded_inputs))
    pred = pred / np.sum(pred)
    pred = np.round(pred, 4)
    return pred[0]
    
