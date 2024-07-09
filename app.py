# Import necessary libraries
import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from underthesea import word_tokenize
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report
from collections import Counter

# Load data
df = pd.read_csv('./data/product_df.csv')
df = df[['Star Rating', 'Comment']]

# Function to label sentiment
def label_sentiment(rating):
    if rating in [1, 2]:
        return '0'
    elif rating == 3:
        return '1'
    elif rating in [4, 5]:
        return '2'
    else:
        return '3'  # Out of range ratings

# Apply sentiment labels to the dataset
df['Sentiment'] = df['Star Rating'].apply(label_sentiment)

# Display first 5 rows with the new sentiment column
print(df.head())

# Data preprocessing
# Check and remove missing values
print(df.isnull().sum())
df = df.dropna(subset=['Comment'])

# Check and remove duplicate data
duplicate_comments = df[df.duplicated(['Comment'])]
print("Duplicate rows in 'Comment' column:")
print(duplicate_comments)
df = df.drop_duplicates(['Comment'])
print("Shape after dropping duplicates:", df.shape)

# Normalize and clean text
def normalize_text(text):
    text = re.sub(r'[^a-zA-ZÀ-ỹà-ỹ0-9\s]', '', text)  # Remove special characters
    text = text.lower()  # Convert to lowercase
    return text

df['Comment'] = df['Comment'].apply(normalize_text)
print(df.head())

# Extract reviews X and labels y
X = df['Comment'].values.tolist()
y = df['Sentiment'].values.tolist()

# Tokenize, build vocabulary, and encode
def tokenize_and_build_vocab_vietnamese(comment):
    tokens = word_tokenize(comment, format="text")
    return tokens.split()

vectorizer = TfidfVectorizer(tokenizer=tokenize_and_build_vocab_vietnamese, token_pattern=None,
                             max_features=5000, ngram_range=(1, 2), max_df=0.85, min_df=5)
X_tfidf = vectorizer.fit_transform(X)
vocabulary = vectorizer.get_feature_names_out()

# Encode labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Balance data using SMOTE
print('Before balancing:', Counter(y))
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_tfidf, y)
print('After balancing:', Counter(y_resampled))
print('Original data size:', X_tfidf.shape)
print('Balanced data size:', X_resampled.shape)

# Split data into train, validation, and test sets
X_train_temp, X_test, y_train_temp, y_test = train_test_split(X_resampled, y_resampled, test_size=0.25, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_temp, y_train_temp, test_size=0.2, random_state=42)

# Convert sparse matrix to array
X_train = X_train.toarray()
X_val = X_val.toarray()
X_test = X_test.toarray()

# Build and train the model
model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(3, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model with early stopping
epochs = 10
batch_size = 64
history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val),
                    callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])

# Evaluate the model on test set
score = model.evaluate(X_test, y_test, verbose=1)
print("Test Score:", score[0])
print("Test Accuracy:", score[1])

# Define preprocessing functions as before
def normalize_text(text):
    return re.sub(r'[^a-zA-ZÀ-ỹà-ỹ0-9\s]', '', text)

def tokenize_and_build_vocab_vietnamese(comment):
    tokens = word_tokenize(comment, format="text")
    return tokens.split()

# Test a new review
new_review = "Shop có cho đổi điện thoại không?"
new_review = normalize_text(new_review)
new_review_tfidf = vectorizer.transform([new_review]).toarray()

prediction = model.predict(new_review_tfidf)
predicted_label = np.argmax(prediction, axis=1)
predicted_sentiment = label_encoder.inverse_transform(predicted_label)
print(f"Review: {new_review}")
print(f"Predicted sentiment: {predicted_sentiment[0]}")

# Classification report
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
print(classification_report(y_test, y_pred_classes, target_names=label_encoder.classes_))

# Save the model
model.save('sentiment_analysis_model.h5')
