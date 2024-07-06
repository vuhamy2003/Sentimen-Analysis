import streamlit as st
import pandas as pd
import numpy as np
import re
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, Embedding, Bidirectional
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import one_hot

# Load dữ liệu
df = pd.read_csv('./data/product_df.csv')
df = df[['Star Rating', 'Comment']]

# Gắn nhãn cảm xúc cho mỗi đánh giá
def label_sentiment(rating):
    if rating in [1, 2]:
        return '0'
    elif rating == 3:
        return '1'
    elif rating in [4, 5]:
        return '2'
    else:
        return '3'  # Nếu có xếp hạng nằm ngoài khoảng 1-5

df['Sentiment'] = df['Star Rating'].apply(label_sentiment)

# Kiểm tra và loại bỏ giá trị khuyết thiếu
df = df.dropna(subset=['Comment'])

# Kiểm tra và loại bỏ dữ liệu trùng lặp
df = df.drop_duplicates(['Comment'])

# Chuẩn hóa và làm sạch văn bản
def remove_special_characters(text):
    return re.sub(r'[^a-zA-ZÀ-ỹà-ỹ0-9\s]', '', text)

def to_lowercase(text):
    return text.lower()

def normalize_text(text):
    text = remove_special_characters(text)
    text = to_lowercase(text)
    return text

df['Comment'] = df['Comment'].apply(normalize_text)

# Mã hóa và chuẩn bị dữ liệu cho mô hình
max_features = 20000 
maxlen = 200
phrases = df['Comment'].tolist()
vocab_size = 50000
encoded_phrases = [one_hot(d, vocab_size) for d in phrases]
df['Comment'] = encoded_phrases
label_encoder = LabelEncoder()
df['Sentiment'] = label_encoder.fit_transform(df['Sentiment'])

# Kiểm tra sự cân bằng của các nhãn
st.write("Phân phối nhãn trong dữ liệu huấn luyện:")
st.write(df['Sentiment'].value_counts())

# Chia dữ liệu thành tập huấn luyện và kiểm tra
X = df['Comment']
y = df['Sentiment']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
x_train = pad_sequences(X_train, maxlen=maxlen)
x_val = pad_sequences(X_val, maxlen=maxlen)

# Xây dựng mô hình nếu chưa có mô hình đã lưu
model_path = 'sentiment_model.h5'

def build_and_train_model():
    model = Sequential()
    inputs = keras.Input(shape=(None,), dtype="int32")
    model.add(inputs)
    model.add(Embedding(50000, 128))
    model.add(Bidirectional(LSTM(64, return_sequences=True)))
    model.add(Bidirectional(LSTM(64)))
    model.add(Dense(3, activation="sigmoid"))
    model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(x_train, y_train, batch_size=32, epochs=30, validation_data=(x_val, y_val))
    model.save(model_path)
    return model

try:
    model = load_model(model_path)
except:
    model = build_and_train_model()

# Giao diện Streamlit
st.title("Dự đoán cảm xúc của đánh giá sản phẩm")

user_input = st.text_area("Nhập đánh giá sản phẩm của bạn:")

if st.button("Dự đoán"):
    if user_input:
        normalized_comment = normalize_text(user_input)
        encoded_comment = one_hot(normalized_comment, vocab_size)
        padded_comment = pad_sequences([encoded_comment], maxlen=maxlen)
        prediction = model.predict(padded_comment)
        sentiment_label = np.argmax(prediction)
        if sentiment_label == 0:
            st.write('Cảm xúc: Tiêu cực')
        elif sentiment_label == 1:
            st.write('Cảm xúc: Trung tính')
        elif sentiment_label == 2:
            st.write('Cảm xúc: Tích cực')
        else:
            st.write('Không xác định')
    else:
        st.write("Vui lòng nhập đánh giá sản phẩm.")

# Đánh giá mô hình trên tập kiểm tra
if st.button("Đánh giá mô hình"):
    _, accuracy = model.evaluate(x_val, y_val)
    st.write(f'Accuracy trên tập kiểm tra: {accuracy:.2f}')
