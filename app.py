import streamlit as st
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the model and vectorizer
model_path = 'trained_news_model.sav'  # Update with your path
vectorizer_path = 'vectorizer.pkl'  # Update with your path

loaded_model = pickle.load(open(model_path, 'rb'))
vectorizer = pickle.load(open(vectorizer_path, 'rb'))

# Streamlit web app title


st.set_page_config(page_title="Fake News Detector", page_icon="📰")
st.title("Welcome to the Fake News Detector Web App 📰🔍")
st.markdown("""
    **Welcome to the Fake News Detector!**  
    Enter news or content in the box below and click the button to check if the news is real or fake 🕵️‍♂️.
    """)
    
# User input text box
input_text = st.text_area("Enter News Content 📰", placeholder="Write the news here...", height=200)

# Check News Button
if st.button("Check News"):
    if input_text:
        # Transform the user input using the vectorizer
        input_data = vectorizer.transform([input_text])
        
        # Get the prediction from the model
        prediction = loaded_model.predict(input_data)
        
        # Show the result with appropriate emojis
        if prediction[0] == 1:
            st.markdown("<h3 style='color: red;'>The news is Fake 🚫</h3>", unsafe_allow_html=True)
        else:
            st.markdown("<h3 style='color: green;'>The news is Real ✅</h3>", unsafe_allow_html=True)
    else:
        st.warning("Please enter some text to check the news.")
st.write("**Note:It is stated that this app can make mistake as the data is old!**")
# Footer section
st.markdown("---")
st.markdown("Made with ❤️ by Subhan Tanveer. For educational purposes.")
