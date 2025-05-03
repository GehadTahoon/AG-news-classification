import streamlit as st
import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Download NLTK data files
nltk.download('stopwords')
nltk.download('punkt')

# Function to preprocess text
def remove_stop_words(txt):
    stop_words = stopwords.words('english')
    return ' '.join([x for x in txt.split() if x not in stop_words])

def remove_punc(txt):
    text_non_punct = "".join([char for char in txt if char not in string.punctuation])
    return text_non_punct

# Load model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained("news_classification_roberta_model")
tokenizer = AutoTokenizer.from_pretrained("news_classification_roberta_model")
# model.eval()

# Title and description
st.title("Text Classification with Transformers")
st.write("Analyze text to predict sentiment.")

user_input = st.text_area("Enter text to classify:", placeholder="Type here...")

user_input = user_input.lower()
remove_stop_words(user_input)
remove_punc(user_input)

labels = ["World", "Sports", "Business", "Sci/Tech"]

if st.button("Classify"):
    if user_input.strip() != "":
        inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            predicted_class_id = torch.argmax(logits, dim=1).item()
        st.success(f"Predicted Class: {labels[predicted_class_id]}")
    else:
        st.warning("Please enter some text.")
