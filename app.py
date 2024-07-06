import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re

# Load the pre-trained model
model_path = 'sentiment_model.h5'
model = load_model(model_path)

# Function to preprocess text
def normalize_text(text):
    text = re.sub(r'[^a-zA-ZÀ-ỹà-ỹ0-9\s]', '', text)
    text = text.lower()
    return text

# Function to predict sentiment
def predict_sentiment(new_comment):
    vocab_size = 50000
    maxlen = 200

    normalized_comment = normalize_text(new_comment)
    encoded_comment = [one_hot(normalized_comment, vocab_size)]
    padded_comment = pad_sequences(encoded_comment, maxlen=maxlen)

    prediction = model.predict(padded_comment)
    sentiment_label = np.argmax(prediction)

    if sentiment_label == 0:
        return 'Tiêu cực'
    elif sentiment_label == 1:
        return 'Trung tính'
    elif sentiment_label == 2:
        return 'Tích cực'
    else:
        return 'Không xác định'

# Streamlit app
st.title('Dự đoán cảm xúc của comment')
comment = st.text_input('Nhập comment:')
if st.button('Dự đoán'):
    if comment:
        sentiment = predict_sentiment(comment)
        st.write('Cảm xúc:', sentiment)
    else:
        st.warning('Vui lòng nhập comment để dự đoán.')
