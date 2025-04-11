import csv
import pandas as pd
from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB

# 1. Lematización del texto con TextBlob
def lematizar_mensaje(mensaje):
    mensaje = mensaje.lower()
    palabras = TextBlob(mensaje).words
    lemas = [palabra.lemmatize() for palabra in palabras]
    return lemas

# 2. Cargar dataset desde archivo
def cargar_dataset(ruta):
    df = pd.read_csv(
        ruta,
        sep='\t',
        quoting=csv.QUOTE_NONE,
        names=["class", "message"]
    )
    print(f"✅ Dataset cargado con {len(df)} mensajes.")
    print(df.groupby("class").count())
    return df

# 3. Entrenar el modelo
def entrenar_modelo(df):
    vectorizador = CountVectorizer(analyzer=lematizar_mensaje)
    X_bow = vectorizador.fit_transform(df["message"])

    tfidf_transformer = TfidfTransformer()
    X_tfidf = tfidf_transformer.fit_transform(X_bow)

    modelo = MultinomialNB()
    modelo.fit(X_tfidf, df["class"])

    return modelo, vectorizador, tfidf_transformer

# 4. Clasificar un mensaje nuevo
def clasificar_mensaje(modelo, vectorizador, transformer, mensaje):
    bow = vectorizador.transform([mensaje])
    tfidf = transformer.transform(bow)
    pred = modelo.predict(tfidf)[0]
    return pred

# 5. Flujo principal
def main():
    df = cargar_dataset("data/SMSSpamCollection")
    modelo, vectorizador, transformer = entrenar_modelo(df)

    print("\n💬 Detector de SPAM (escribí un mensaje o 'salir')")
    while True:
        mensaje = input("➡️ Ingrese un mensaje: ")
        if mensaje.lower() == "salir":
            print("👋 Finalizando...")
            break
        resultado = clasificar_mensaje(modelo, vectorizador, transformer, mensaje)
        print(f"📌 Resultado: {resultado.upper()}")

if __name__ == "__main__":
    main()
