import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from builder import limpiar_texto, vectorizar_texto

df = pd.read_csv("data/SMSSpamCollection", sep='\t', header=None, names=["label", "message"])

df["message_clean"] = df["message"].apply(limpiar_texto)
X, vectorizador = vectorizar_texto(df["message_clean"])
y = df["label"].map({"ham": 0, "spam": 1})

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
modelo = MultinomialNB()
modelo.fit(X_train, y_train)

joblib.dump(modelo, "models/modelo_spam.pkl")
joblib.dump(vectorizador, "models/vectorizador.pkl")
print("✅ Modelo y vectorizador guardados.")