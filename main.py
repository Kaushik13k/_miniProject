import streamlit as st
import streamlit as st
import spacy_streamlit
import spacy
from PIL import Image

nlp = spacy.load('en_core_web_sm')

# NLP PACKAGES
import spacy
from textblob import TextBlob
from gensim.summarization import summarize
# sumy packages
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer

import re
import streamlit as st
from pages.fetch import *
import sys

from googletrans import Translator

st.sidebar.title("About The App")
st.sidebar.write(
    "NLP is a subfield of computer science and artificial intelligence concerned with interactions between computers and human (natural) languages. It is used to apply machine learning algorithms to text and speech.we can use NLP to create systems like text summarization, named entity recognition,Sentence tokenization,sentiment")
img = Image.open("head.jpeg")
st.image(img, width=800, )
imge = Image.open("mainnlp.JPG")
st.sidebar.image(imge, width=400, )

MENU = ["Home", "NER"]
choice = st.sidebar.selectbox("MENU", MENU)


# summary Fxn
def sumy_summarizer(docx):
    parser = PlaintextParser.from_string(docx, Tokenizer("english"))


    lex_summarizer = LexRankSummarizer()
    summary = lex_summarizer(parser.document, 3)
    summary_list = [str(sentence) for sentence in summary]
    result = ' '.join(summary_list)
    return result


# other packages

def main():


    st.title("NATURAL LANGUAGE PROCESSING WEB APPLICATION")
    st.subheader("Choose the type of NLP service you like to use:")

# Tokenization

# Named Entity


if choice == "Home":
    st.subheader("Tokenization")
    raw_text = st.text_area("your text", "Enter text here")
    docx = nlp(raw_text)
if st.button("Tokenize"):
    spacy_streamlit.visualize_tokens(docx, attrs=['text', 'lemma_', 'pos_'])



elif choice == "NER":
    st.subheader("Named Entity Recognition")
    raw_text = st.text_area("your text", "Enter text here")
    docx = nlp(raw_text)
    spacy_streamlit.visualize_ner(docx, labels=nlp.get_pipe('ner').labels)

# Sentiment Analysis
if st.checkbox("Show Sentiment Analysis"):
    st.subheader("Sentiment of Your Text")
    message = st.text_area("Enter Your Text", "Type Here")
if st.button("analyse"):
    blob = TextBlob(message)
    result_sentiment = blob.sentiment
    st.success(result_sentiment)
    st.area_chart(result_sentiment)

# Text Summarization
if st.checkbox("Show Text Summarization"):
    st.subheader("Summarize Your Text")
    message = st.text_area("Enter Your Text", "Give your paragraph")
    summary_options = st.selectbox("Choice Your Summarizer", ("gensim", "sumy"))
if st.button("Summarize"):
    if summary_options == 'gensim':
        st.text("Using Gensim..")
        summary_result = summarize(message)
    elif summary_options == 'sumy':
        st.text("using Sumy..")
        summary_result = sumy_summarizer(message)
    else:
        st.warning("Using Default Summarizer")
        st.text("using Gensim")
        summary_result = summarize(message)
        st.success(summary_result)

if __name__ == '__main__':
    main()

