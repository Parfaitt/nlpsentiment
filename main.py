import streamlit as st
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
import string
nltk.download('punkt')
nltk.download('wordnet')

# Configuration
st.set_page_config(page_title="Analysedessentiments", page_icon=":bar_chart:", layout="wide")

# Header
st.title(":bar_chart: ANALYSE DES SENTIMENTS ")
st.write('Application de comportements clients')
st.markdown('Auteur: Parfait Ngoran')
st.markdown('email: parfaittanoh42@gmail.com')
st.markdown('<style>div.block-container{padding-top:1rem;}</style>', unsafe_allow_html=True)

dictionary = {0: "Negative", 1: "Positive"}

lemmer = nltk.stem.WordNetLemmatizer()

def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]

remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)

def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

st.sidebar.header("Informations")
st.sidebar.write('''
# Application prédiction des comportement des clients
Cet ensemble de données est utilisé pour prédire le comportement des clients

Auteur: Parfait Tanoh N'goran
''')

review = st.text_area("Votre avis compte, Qu'en pensez-vous du film ?")

# Ajout du bouton de prédiction
if st.button("Predict"):
    if review:
        tfidfvectorizer = pickle.load(open('tfidfvectorizer_19k.pkl', 'rb'))
        x_transform = tfidfvectorizer.transform([review])
        x_transform = x_transform.toarray()
        model = pickle.load(open('model_sgd.pkl', 'rb'))
        hasil = model.predict(x_transform)

        st.write(f"Sentiment: {dictionary[hasil[0]]}")
        st.write(f"Review: {review}")
