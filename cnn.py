import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout
from keras.callbacks import EarlyStopping
from keras.regularizers import l2
from sklearn.metrics import classification_report
from gensim.models import KeyedVectors
import random

# Load pre-trained Word2Vec model
word2vec = KeyedVectors.load_word2vec_format('word2vec_vi_words_100dims.txt', binary=False)

# Load data
df = pd.read_csv('./data/product_df.csv')
df = df[['Star Rating', 'Comment']]

def label_sentiment(rating):
    if rating in [1, 2]:
        return '0'
    elif rating == 3:
        return '1'
    elif rating in [4, 5]:
        return '2'
    else:
        return '3'

df['Sentiment'] = df['Star Rating'].apply(label_sentiment)

# Data preprocessing
df = df.dropna(subset=['Comment'])
df = df.drop_duplicates(['Comment'])

def remove_special_characters(text):
    return re.sub(r'[^a-zA-ZÀ-ỹà-ỹ0-9\s]', '', text)

def to_lowercase(text):
    return text.lower()

def normalize_text(text):
    text = remove_special_characters(text)
    text = to_lowercase(text)
    return text

df['Comment'] = df['Comment'].apply(normalize_text)

# Data Augmentation: Synonym Replacement
def synonym_replacement(comment, n):
    words = comment.split()
    new_words = words.copy()
    random_word_list = list(set([word for word in words if word in word2vec.key_to_index]))
    random.shuffle(random_word_list)
    num_replaced = 0
    for random_word in random_word_list:
        synonyms = word2vec.most_similar(random_word, topn=10)
        for synonym, _ in synonyms:
            if synonym not in new_words:
                new_words = [synonym if word == random_word else word for word in new_words]
                num_replaced += 1
                if num_replaced >= n:
                    break
        if num_replaced >= n:
            break
    sentence = ' '.join(new_words)
    return sentence

augmented_comments = df['Comment'].apply(lambda x: synonym_replacement(x, 2))
df = df.append(pd.DataFrame({'Comment': augmented_comments, 'Star Rating': df['Star Rating'], 'Sentiment': df['Sentiment']}))

# Tokenization
df['Tokenized_Comment'] = df['Comment'].apply(lambda x: word_tokenize(x))

# TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer(tokenizer=lambda x: x, lowercase=False)
tfidf_matrix = tfidf_vectorizer.fit_transform(df['Tokenized_Comment'].tolist())

# Prepare data for the model
max_length = 40
embedding_dim = 100  # Dimension of the Word2Vec embeddings
vocab_size = len(tfidf_vectorizer.vocabulary_)

X = pad_sequences(tfidf_matrix.toarray(), maxlen=max_length)
y = pd.get_dummies(df['Sentiment']).values

# Create embedding matrix
embedding_matrix = np.zeros((vocab_size, embedding_dim))
for word, i in tfidf_vectorizer.vocabulary_.items():
    if word in word2vec.key_to_index:
        embedding_matrix[i] = word2vec[word]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build CNN model
model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length, weights=[embedding_matrix], trainable=False))
model.add(Conv1D(filters=128, kernel_size=5, activation='relu'))
model.add(GlobalMaxPooling1D())
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu', kernel_regularizer=l2(0.01)))
model.add(Dense(3, activation='softmax'))

# Compile model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Early Stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=3)

# Train model
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stopping])

# Plot learning curves
plt.plot(history.history['accuracy'], label='Training accuracy')
plt.plot(history.history['val_accuracy'], label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Evaluate model
y_pred = model.predict(X_test)
y_pred_labels = np.argmax(y_pred, axis=1)
y_test_labels = np.argmax(y_test, axis=1)
print(classification_report(y_test_labels, y_pred_labels))
