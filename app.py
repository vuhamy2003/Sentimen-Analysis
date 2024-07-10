import streamlit as st
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from underthesea import word_tokenize
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping 
from sklearn.model_selection import train_test_split

@st.cache_data()
def load_data():
    df = pd.read_csv('./data/product_df.csv')
    df = df[['Star Rating', 'Comment']]
    return df

def label_sentiment(rating):
    if rating in [1, 2]:
        return 'Tiêu cực'
    elif rating == 3:
        return 'Trung tính'
    elif rating in [4, 5]:
        return 'Tích cực'
    else:
        return 'Không xác định'

def preprocess_data(df):
    df['Sentiment'] = df['Star Rating'].apply(label_sentiment)
    df = df.dropna(subset=['Comment'])
    df = df.drop_duplicates(['Comment'])
    df['Comment'] = df['Comment'].apply(normalize_text)
    return df

def remove_special_characters(text):
    return re.sub(r'[^a-zA-ZÀ-ỹà-ỹ0-9\s]', '', text)

def to_lowercase(text):
    return text.lower()

def normalize_text(text):
    text = remove_special_characters(text)
    text = to_lowercase(text)
    return text

def tokenize_and_build_vocab_vietnamese(comment):
    tokens = word_tokenize(comment, format="text")
    return tokens.split()

def train_model(X, y):
    vectorizer = TfidfVectorizer(tokenizer=tokenize_and_build_vocab_vietnamese, token_pattern=None,
                                 max_features=5000, ngram_range=(1, 2), max_df=0.85, min_df=5)
    X_tfidf = vectorizer.fit_transform(X)
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_tfidf, y_encoded)

    X_train_temp, X_test, y_train_temp, y_test = train_test_split(X_resampled, y_resampled, test_size=0.25, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train_temp, y_train_temp, test_size=0.2, random_state=42)

    model = Sequential()
    model.add(Dense(128, activation='relu', input_shape=(X_train.shape[1],)))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(3, activation='softmax'))

    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.fit(X_train.toarray(), y_train, epochs=10, batch_size=64, validation_data=(X_val.toarray(), y_val),
              callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])
    
    return model, vectorizer, label_encoder

# Giao diện Streamlit app
st.title("Sentiment Analysis cho đánh giá điện thoại của TGDD")
st.write("Nhập vào đánh giá để phân loại:")

df = load_data()
df = preprocess_data(df)

X = df['Comment'].values.tolist()
y = df['Sentiment'].values.tolist()

model, vectorizer, label_encoder = train_model(X, y)

user_input = st.text_area("Nhập đánh giá sản phẩm (mỗi đánh giá một dòng)", value='', height=200)

if st.button("Dự đoán"):
    if user_input:
        # Chia input thành các đánh giá khi xuống dòng
        user_reviews = user_input.split('\n')
        results = []

        for review in user_reviews:
            if review.strip():  # Kiểm tra xem nếu đánh giá không rỗng
                user_input_processed = normalize_text(review)
                user_input_tfidf = vectorizer.transform([user_input_processed]).toarray()

                prediction = model.predict(user_input_tfidf)
                predicted_label = np.argmax(prediction, axis=1)
                predicted_sentiment = label_encoder.inverse_transform(predicted_label)

                results.append(f"**Đánh giá:** {review}")
                results.append(f"**Dự đoán đánh giá:** {predicted_sentiment[0]}")
                results.append("")  # Dòng trống phân tách

        # Kết hợp các kết quả với '\n' để đảm bảo mỗi output xuất hiện trên một dòng mới
        st.markdown('\n'.join(results))
    else:
        st.write("Hãy nhập đánh giá vào đây để dự đoán.")
