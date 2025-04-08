from sklearn.datasets import load_digits
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler

# Cargar el dataset de dígitos
digits = load_digits()
X = digits.data
y = digits.target


# 70% de los datos para entrenamiento y 30% para prueba
X_train, X_test, Y_train, Y_test = train_test_split(X, y,  
                                                    test_size = 0.3, 
                                                    random_state = 42, 
                                                    stratify = y)

# Estandarización de los datos
sc = StandardScaler()
sc.fit(X_train)

X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

# configuración del árbol de decisión
arbol = DecisionTreeClassifier(criterion='entropy',
                                max_depth=8, random_state=1)

# ajustar el modelo
arbol.fit(X_train_std, Y_train)

# predicción de clasificación
Y_pred = arbol.predict(X_test_std)

# evaluación del modelo
accuracy = accuracy_score(Y_test, Y_pred)
print(f"Exactitud (accuracy) del modelo: {accuracy * 100:.2f}%")
print('\ninforme de clasificación')
print(classification_report(Y_test, Y_pred))