from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# Cargar el dataset de dígitos
digits = load_digits()


X = digits.data  
y = digits.target

# Dividir los datos en dos conjuntos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Configuracion random Forest
clf = RandomForestClassifier(n_estimators=100, random_state=42)

# Entrenar el modelo
clf.fit(X_train, y_train)

# Predecir en el conjunto de prueba
y_pred = clf.predict(X_test)

# Evaluar el modelo
accuracy = accuracy_score(y_test, y_pred)


print(f"Exactitud (accuracy) del modelo: {accuracy * 100:.2f}%")
print("\nInforme de clasificación:")
print(classification_report(y_test, y_pred))
