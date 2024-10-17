import requests
from bs4 import BeautifulSoup
import numpy as np 
import pandas as pd 
import streamlit as st  
# import spacy
import matplotlib.pyplot as plt
from predict import predict_sentiments
import re

# nlp = spacy.load('en_core_web_sm')

# Function to refine the text from the URL
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
    tokens = chunk_long_sentence(clean_text)
    for sent in tokens.sents:
        if len(sentence) > 100:
            chunks = chunk_long_sentence(sentence)
            for c in chunks:
                if len(c) > 100:
                    k=c.text.strip()
                    sentence.append(k[:-100])
        sentence.append((sent.text.strip()))
    
    return sentence

# Streamlit app
st.title("Sentiment Analysis App")


input_type = st.radio("Choose input type:", ("News Link","Text"))
sentence=[]
if input_type == "News Link":
    link = st.text_input("Enter the news link:")
    if link:
        sentence = refine(link)
        

elif input_type == "Text":
    user_input = st.text_area("Enter your text:")
    sentence = user_input.split('.')

positive_count = 0
negative_count = 0
print(len(sentence))
if len(sentence)>1:
    sentiment = []
    for s in sentence:
        print(rspace(s))
        out = predict_sentiments(rspace(s)) 
        predictions=np.argmax(out)
        if predictions == 0:
            sentiment.append([s, "Negative"])
        else:
            sentiment.append([s, "Positive"])

        positive_count =positive_count + ((len(s))*float(out[1]))
        negative_count= negative_count+ (len(s)*float(out[0]))
            

    # Create a DataFrame to display results
    df = pd.DataFrame(sentiment, columns=['Sentence', 'Sentiment'])
    st.write(df)
    sentiment_count = df['Sentiment'].value_counts()
    positive_sen = sentiment_count.get('Positive', 0)
    negative_sen = sentiment_count.get('Negative', 0)
    
    st.write(f"Positive Sentence: {positive_sen}      Negative Sentence: {negative_sen}")
    if(positive_count>negative_count):
        st.write(f"Overall: Positive")
    elif(positive_count<negative_count):
        st.write(f"Overall: Negative")
    else:
        st.write(f"Overall: Neutral")
    st.write(f"Positive Sentence: {positive_count}      Negative Sentence: {negative_count}")
    labels = ['Positive', 'Negative']
    sizes = [positive_count, negative_count]
    colors = ['#66b3ff', '#ff6666']

    fig, ax = plt.subplots()

    # Create the pie chart
    ax.pie(sizes, colors=colors, autopct='%1.1f%%', startangle=90)
    ax.axis('equal')  

    ax.legend(labels, loc='center', bbox_to_anchor=(0.5, -0.1), ncol=2, fontsize='medium', title='Labels', title_fontsize='medium')

    # Adjust layout to make room for the legend
    plt.subplots_adjust(bottom=0.3)

    # Display the chart in Streamlit
    st.pyplot(fig)


