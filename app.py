import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string

nltk.download('punkt_tab')
nltk.download('stopwords')

@st.cache_resource
def load_model():
    with open('vectorizer.pkl', 'rb') as f:
        tv = pickle.load(f)
    
    with open('model.pkl', 'rb') as f:
        mnb = pickle.load(f)
    
    return tv, mnb

# Initialize
tv, mnb = load_model()
ps = PorterStemmer()

# Transform text function
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    
    text = y[:]
    y.clear()
    
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    
    text = y[:]
    y.clear()
    
    for i in text:
        y.append(ps.stem(i))
    
    return " ".join(y)

st.set_page_config(page_title="SMS Spam Classifier", layout="centered")

st.title("📧 SMS Spam Classification")
st.write("Enter a message to check if it's spam or ham (legitimate)")

# Input text area
user_input = st.text_area("Enter your message:", placeholder="Type a message here...", height=100)

# Classify button
if st.button("🔍 Classify", use_container_width=True):
    if user_input.strip():
        # Transform and predict
        transformed_text = transform_text(user_input)
        vectorized_text = tv.transform([transformed_text]).toarray()
        prediction = mnb.predict(vectorized_text)[0]
        
        # Display result
        if prediction == 0:
            st.success("✅ **Ham** - This is a legitimate message", icon="✅")
        else:
            st.error("⚠️ **Spam** - This appears to be spam", icon="⚠️")
    else:
        st.warning("Please enter a message to classify!")

# Footer
st.divider()
st.caption("SMS Spam Classifier | Built with Streamlit & Scikit-learn") 