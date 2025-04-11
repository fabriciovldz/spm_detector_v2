import joblib
from builder import limpiar_texto

# Cargar modelo y vectorizador
modelo = joblib.load("models/modelo_spam.pkl")
vectorizador = joblib.load("models/vectorizador.pkl")

def clasificar(texto):
    texto_limpio = limpiar_texto(texto)
    texto_vectorizado = vectorizador.transform([texto_limpio])
    resultado = modelo.predict(texto_vectorizado)[0]
    return "SPAM" if resultado == 1 else "NO SPAM"

# Interfaz terminal
print("💬 Detector de SPAM (escribí un mensaje o 'salir')")

while True:
    entrada = input("➡️ Mensaje: ")
    if entrada.lower() == "salir":
        break
    print("📌 Resultado:", clasificar(entrada))
