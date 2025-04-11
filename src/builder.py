import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('stopwords')
STOPWORDS = set(stopwords.words('spanish'))

def limpiar_texto(texto):
    texto = texto.lower()
    texto = re.sub(r'\W', ' ', texto)
    texto = re.sub(r'\s+', ' ', texto)
    texto = ' '.join([t for t in texto.split() if t not in STOPWORDS])
    return texto

def vectorizar_texto(corpus):
    vectorizador = TfidfVectorizer()
    X = vectorizador.fit_transform(corpus)
    return X, vectorizador