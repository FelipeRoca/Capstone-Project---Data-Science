import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report

# --- PREPROCESAMIENTO DE CLIENTES ---
clientes = pd.read_csv('clientes.csv')
destinatarios = pd.read_csv('Destinatarios.txt', encoding='ISO-8859-1')
mercados = pd.read_excel("datosMercados.xlsx")


#eliminamos columnas que no usamos
clientes.drop("IdCliente" , axis=1, inplace=True)
clientes.drop("Nombre" , axis=1, inplace=True)
clientes.drop("Apellido" , axis=1, inplace=True)
clientes.drop("Telefono" , axis=1, inplace=True)
clientes.drop("Direccion" , axis=1, inplace=True)
clientes.drop("Propietario" , axis=1, inplace=True)
clientes.drop("Email" , axis=1, inplace=True)
clientes.drop("FechaNacimiento", axis=1, inplace=True)
clientes.drop("FechaPrimeraCompra", axis=1, inplace=True)  #Eliminamos por el error al ejecutar. Borrar linea en un futuro y tratar el dato

# INFORMACION PARA SABER COMO SE COMPONEN LAS TABLAS
# clientes.info()
# destinatarios.info()
# mercados.info()



# Poner valores faltantes en IngresoAnual con la mediana
imputer_ingresos = SimpleImputer(strategy='median')
clientes['IngresoAnual'] = imputer_ingresos.fit_transform(clientes[['IngresoAnual']])

# Verificar si quedan valores nulos
#print(clientes.isnull().sum())



# ver valores unicos en la columna Distancia
# print(clientes['Distancia'].unique())

clientes['Distancia'] = clientes['Distancia'].str.replace("'", "") # eliminar comillas 
clientes['Distancia'] = clientes['Distancia'].str.replace(" Km.", "") # eliminar " Km."


def convertir_distancia(distancia):
    if distancia == '10+':
        return 12 
    elif distancia == '2-5':
        return 3.5
    elif distancia == '1-2':
        return 1.5
    elif distancia == '0-1':
        return 0.5
    elif distancia == '5-10':
        return 7.5
    return np.nan # para cualquier otro valor


clientes['Distancia'] = clientes['Distancia'].apply(convertir_distancia)

# imputar los NaN
imputer_distancia = SimpleImputer(strategy='median')
clientes['Distancia'] = imputer_distancia.fit_transform(clientes[['Distancia']])

# verificar tipo de dato ahora
# print(clientes['Distancia'].dtype)
# print(clientes['Distancia'].unique())



# print(clientes.head())

label_encoders = {}
for column in ['EstadoCivil', 'Genero', 'Educacion', 'Ocupacion', 'Region']:
    le = LabelEncoder()
    clientes[column] = le.fit_transform(clientes[column])
    label_encoders[column] = le # Guardar los encoders para usar luego en destinatarios

# print(clientes.head())
# Genero (M->1 , F->0)
# EstadoCivil (Casado->0, Soltero->1)
# Educacion (Secundario-> 1 , Postgrado-> 2 , Estudios universitarios(en curso)->3 , Licenciatura->4)
# Ocupacion (Obrero especializado -> 0 , ->1 , Gestion-> 2 , Obrero - > 3 , profesional -> 4)
# Region (Centro-> 0 , -> 1 , Norte-> 2 , Sur->3)



# Variables mas importantes
y = clientes['ComproBicicleta']
X = clientes.drop('ComproBicicleta', axis=1)

# Divido entre entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# entreno arbol de decision    
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)


# evaluo el modelo
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))


# DATOS IMPORTANTES DE ESTA EVALUACION
# Recordar que 0 son lo que no compraron bicis y 1 los que si lo hicieron
# precision es de todas las instancias que el modelo predijo como pertenecientes a una clase, ¿qué proporción fue realmente correcta? -> para 0: 79% realmente no lo hicieron, para 1: 65 % realmente lo hicieron
# recall indica de todas las instancias que realmente pertenecen a una clase, ¿qué proporción fue correctamente identificada por el modelo?
# F1-score es la media armónica ponderada de la precisión y el recall. Proporciona una medida única que equilibra ambos aspectos
# Support es el número de instancias reales que pertenecen a cada clase en el conjunto de prueba
# Accuracy es la proporción de todas las predicciones (tanto para la clase 0 como para la clase 1) que fueron correctas. En este caso, el modelo tuvo una precisión general del 73%
# Macro avg es el promedio simple de la precisión, el recall y el f1-score entre las dos clases
# Weighted avg: Es el promedio ponderado de la precisión, el recall y el f1-score entre las dos clases, donde el peso es el soporte (el número de instancias reales en cada clase)