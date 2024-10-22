import requests
from bs4 import BeautifulSoup
import numpy as np 
import pandas as pd 
import streamlit as st  

import matplotlib.pyplot as plt
from predict import predict_sentiments
import re


def adds(text):
    return re.sub(r'(?<!^)(?<![A-Z])(?=[A-Z])', ' ', text)


def chunk_long_sentence(sentence):
    chunks = re.split(r'(?<=[.!?]) +', sentence)  # Split at punctuation
    return chunks


def rspace(sentence):
    return re.sub(r'\s+', ' ', sentence).strip()


def refine(link):
    req = requests.get(link)
    req.encoding = 'utf-8'
    html = req.text
    soup = BeautifulSoup(html, 'html.parser')
    text = soup.get_text()
    
    clean_text = text.replace("\n", " ")
    clean_text = clean_text.replace("/", " ")       
    clean_text = ''.join([c for c in clean_text if c != "'"])
    
    sentence = []
    clean_text = adds(clean_text)
    tokens = chunk_long_sentence(clean_text)
    for sent in tokens:
        if len(sentence) > 100:
            chunks = chunk_long_sentence(sentence)
            for c in chunks:
                if len(c) > 100:
                    k = c.strip()
                    sentence.append(k[:-100])
        sentence.append(sent.strip())
    
    return sentence


# Streamlit app
st.set_page_config(
    page_title="Sentiment Analysis App",
    page_icon="📝",
    layout="wide",
)

st.title("📝 Sentiment Analysis App")
st.markdown("""
This app analyzes the sentiment of your input text or a news article URL. 
Choose your input method and see the sentiment distribution!
""")

# Sidebar for inputs
st.sidebar.header("Input Options")

input_type = st.sidebar.radio("Choose input type:", ("News Link", "Text"))

sentence = []
if input_type == "News Link":
    link = st.sidebar.text_input("Enter the news link:")
    if link:
        with st.spinner("Fetching and processing the article..."):
            sentence = refine(link)
        st.sidebar.success("Article processed successfully!")
elif input_type == "Text":
    user_input = st.sidebar.text_area("Enter your text:", height=200)
    if user_input:
        with st.spinner("Processing your text..."):
            sentence = user_input.split('.')
        st.sidebar.success("Text processed successfully!")

if sentence:
    with st.spinner("Analyzing sentiments..."):
        positive_count = 0
        negative_count = 0
        sentiment = []
        for s in sentence:
            cleaned_sentence = rspace(s)
            if cleaned_sentence:  # Ensure sentence is not empty
                out = predict_sentiments(cleaned_sentence) 
                predictions = np.argmax(out)
                sentiment_label = "Negative" if predictions == 0 else "Positive"
                sentiment.append([cleaned_sentence, sentiment_label])

                positive_count += len(s) * float(out[1])
                negative_count += len(s) * float(out[0])
        
        # Create a DataFrame to display results
        df = pd.DataFrame(sentiment, columns=['Sentence', 'Sentiment'])
    
    st.success("Sentiment analysis completed!")
    
    # Display Sentiment Breakdown
    st.subheader("Sentiment Breakdown")
    sentiment_count = df['Sentiment'].value_counts()
    cols = st.columns(2)
    cols[0].metric("Positive Sentences", sentiment_count.get('Positive', 0))
    cols[1].metric("Negative Sentences", sentiment_count.get('Negative', 0))

    # Overall Sentiment
    if positive_count > negative_count:
        overall = "Positive 👍"
    elif positive_count < negative_count:
        overall = "Negative 👎"
    else:
        overall = "Neutral 😐"
    
    st.markdown(f"### **Overall Sentiment: {overall}**")
    
    # Display Numerical Sentiment Scores
    st.subheader("Sentiment Scores")
    score_cols = st.columns(2)
    score_cols[0].metric("Positive Score", f"{positive_count:.2f}")
    score_cols[1].metric("Negative Score", f"{negative_count:.2f}")

    # Pie Chart Visualization
    st.subheader("Sentiment Distribution")
    labels = ['Positive', 'Negative']
    sizes = [positive_count, negative_count]
    colors = ['#66b3ff', '#ff6666']

    fig, ax = plt.subplots()
    ax.pie(sizes, colors=colors, autopct='%1.1f%%', startangle=90)
    ax.axis('equal')  

    ax.legend(labels, loc='center', bbox_to_anchor=(0.5, -0.1), ncol=2, fontsize='medium', title='Labels', title_fontsize='medium')

    plt.subplots_adjust(bottom=0.3)

    st.pyplot(fig)

    # Display Detailed Sentiment Table
    with st.expander("View Detailed Sentiment Analysis"):
        st.dataframe(df)

    # Download Button for Results
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Sentiment Analysis CSV",
        data=csv,
        file_name='sentiment_analysis.csv',
        mime='text/csv',
    )


