from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# Cargar el dataset de dígitos
digits = load_digits()
X = digits.data
y = digits.target

# Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Crear y entrenar el modelo de regresión logística
clf = LogisticRegression(max_iter=10000)  # Aumentar max_iter para asegurar convergencia

#Entrenar el modelo
clf.fit(X_train, y_train)

# Predecir en el conjunto de prueba
y_pred = clf.predict(X_test)

# Evaluar el modelo
accuracy = accuracy_score(y_test, y_pred)
print(f"Exactitud (accuracy) del modelo: {accuracy * 100:.2f}%")
print("\nInforme de clasificación:")
print(classification_report(y_test, y_pred))
