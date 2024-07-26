import numpy as np
import nltk
import tensorflow as tf
from tensorflow.keras.models import load_model
import streamlit as st
from nltk.tokenize import word_tokenize
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import SimpleRNN, Dense, Dropout, Embedding
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import one_hot
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re

nltk.download('stopwords')
nltk.download('wordnet')
leme = WordNetLemmatizer()
voc_size = 3000

model = load_model('beginner.h5')


def preprocess(text):
    review = re.sub('[^a-zA-Z]', ' ', text.lower())  # Combine steps
    review = word_tokenize(review)
    review = [word for word in review if not word in stopwords.words('english')]
    review = [leme.lemmatize(word) for word in review]
    review = ' '.join(review)
    onehot_repr = one_hot(review, voc_size)  # One-hot encode directly
    embedded_docs = pad_sequences([onehot_repr], padding='post', maxlen=50)
    return np.array(embedded_docs)


st.title('Positive-Negative Sentiment Analysis')
st.write('Enter Text.')

# User input
user_input = st.text_area('Text')

if st.button('Classify'):
    preprocessed_input = preprocess(user_input)

    ## Make prediction
    prediction = model.predict(preprocessed_input)
    sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'

    # Display the result
    st.write(f'Sentiment: {sentiment}')
    st.write(f'Prediction Score: {prediction[0][0]}')
else:
    st.write('Enter Text.')
