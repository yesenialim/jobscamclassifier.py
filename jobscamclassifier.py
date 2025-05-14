import pandas as pd
import string
import re
from sklearn.feature_extraction.text import TfidfVectorizer

# Load dataset
df = pd.read_csv('/Users/yesenia/Downloads/fake_job_postings.csv')

# Combine useful text columns
df['text'] = df[['title', 'description', 'requirements', 'company_profile']].fillna('').agg(' '.join, axis=1)

# Keep only necessary columns
df = df[['text', 'fraudulent']]

# Optional: downsample real jobs to balance dataset
real = df[df['fraudulent'] == 0]
fake = df[df['fraudulent'] == 1]


def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
    text = re.sub(r'\@w+|\#','', text)
    text = re.sub(r'[^A-Za-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df['clean_text'] = df['text'].apply(clean_text)

# Convert text to numeric vectors using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X = vectorizer.fit_transform(df['clean_text'])
y = df['fraudulent']

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

def scam_type(text):
    text = text.lower()
    if 'crypto' in text:
        return 'Crypto'
    elif 'data entry' in text or 'form filling' in text:
        return 'Data Entry'
    elif 'customer support' in text:
        return 'Customer Support'
    else:
        return 'Other'

df['scam_type'] = df['clean_text'].apply(scam_type)

# streamlit_app.py
import streamlit as st

st.title("Fake Job Posting Detector")
job_text = st.text_area("Paste the job description:")

if st.button("Check"):
    cleaned = clean_text(job_text)
    vect = vectorizer.transform([cleaned])
    result = model.predict(vect)
    scam = "⚠️ FAKE" if result[0] == 1 else "✅ Likely Real"
    st.subheader(scam)
