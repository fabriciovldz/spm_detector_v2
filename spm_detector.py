import csv
import pandas as pd
from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB


def lematizar_mensaje(mensaje):
    mensaje = mensaje.lower()
    palabras = TextBlob(mensaje).words
    lemas = [palabra.lemmatize() for palabra in palabras]
    return lemas


def cargar_dataset(ruta):
    df = pd.read_csv(
        ruta,
        sep='\t',
        quoting=csv.QUOTE_NONE,
        names=["class", "message"]
    )
    print(f" Dataset loaded with {len(df)} messages.")
    print(df.groupby("class").count())
    return df


def entrenar_modelo(df):
    vectorizador = CountVectorizer(analyzer=lematizar_mensaje)
    X_bow = vectorizador.fit_transform(df["message"])

    tfidf_transformer = TfidfTransformer()
    X_tfidf = tfidf_transformer.fit_transform(X_bow)

    modelo = MultinomialNB()
    modelo.fit(X_tfidf, df["class"])

    return modelo, vectorizador, tfidf_transformer


def clasificar_mensaje(modelo, vectorizador, transformer, mensaje):
    bow = vectorizador.transform([mensaje])
    tfidf = transformer.transform(bow)
    pred = modelo.predict(tfidf)[0]
    return pred


def main():
    df = cargar_dataset("data/SMSSpamCollection")
    modelo, vectorizador, transformer = entrenar_modelo(df)

    print("\n input or 'exit')")
    while True:
        mensaje = input("input: ")
        if mensaje.lower() == "exit":
            print("chau.")
            break
        resultado = clasificar_mensaje(modelo, vectorizador, transformer, mensaje)
        print(f"result: {resultado.upper()}")

if __name__ == "__main__":
    main()