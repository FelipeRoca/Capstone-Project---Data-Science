import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import seaborn as sns


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

# para reemplazar 2 por 1
clientes['EstadoCivil'] = clientes['EstadoCivil'].replace(2, 0)  #ver a que se debe que haya estados civiles 2. Y ver si eliminarlos o modificar esto
# los valores estadisticos no varian


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



#   GRAFICAS DE ESTUDIO DE CLIENTES

# estado civil vs compra bicis
plt.subplot(2, 2, 1)
sns.countplot(x='EstadoCivil', hue='ComproBicicleta', data=clientes)
plt.title('Estado Civil vs Compro Bicicleta')
plt.xlabel('Estado Civil (0=Casado, 1=Soltero)')
plt.ylabel('Cantidad')


# hijos vs compra bici
plt.subplot(2, 2, 2)
sns.countplot(x='TotalHijos', hue='ComproBicicleta', data=clientes)
plt.title('Total de Hijos vs Compro Bicicleta')
plt.xlabel('Cantidad de Hijos')
plt.ylabel('Cantidad')

# cantidad de autos vs Compra bici
plt.subplot(2, 2, 3)
sns.countplot(x='CantAutomoviles', hue='ComproBicicleta', data=clientes)
plt.title('Cantidad de Automóviles vs Compro Bicicleta')
plt.xlabel('Cantidad de Automóviles')
plt.ylabel('Cantidad')

# distancia al trabajo vs compra bicis
plt.subplot(2, 2, 4)
# redondeamos p agrupar por distancia entera
clientes['DistanciaRedondeada'] = clientes['Distancia'].round()
sns.countplot(x='DistanciaRedondeada', hue='ComproBicicleta', data=clientes)
plt.title('Distancia al trabajo vs Compro Bicicleta')
plt.xlabel('Distancia al trabajo (km)')
plt.ylabel('Cantidad')

# grafico matriz correlacion
plt.figure(figsize=(12, 8))
sns.heatmap(clientes.corr(), annot=True, cmap='coolwarm')
plt.title('Matriz de Correlación entre Variables Numéricas')






# tasa de compra por grupo en eatado civil
conversion_estado = clientes.groupby('EstadoCivil')['ComproBicicleta'].mean()
print(conversion_estado)
# el 37% de los solteros compro bicicletas
# el 42% de los casados compra bicicletas
plt.figure(figsize=(10, 6))
sns.barplot(x=conversion_estado.index, y=conversion_estado.values)
plt.title('Tasa de compra por estado civil')
plt.xlabel('Estado Civil')
plt.ylabel('Tasa de compra')



# tasa de compra por grupo en hijos
conversion_hijos = clientes.groupby('TotalHijos')['ComproBicicleta'].mean()
print(conversion_hijos)

plt.figure(figsize=(10, 6))
sns.barplot(x=conversion_hijos.index, y=conversion_hijos.values)
plt.title('Tasa de compra por cantidad de hijos')
plt.xlabel('Cantidad de hijos')
plt.ylabel('Tasa de compra')



# tasa de compra por grupo en cantidad de autos
conversion_autos = clientes.groupby('CantAutomoviles')['ComproBicicleta'].mean()
print(conversion_autos)

plt.figure(figsize=(10, 6))
sns.barplot(x=conversion_autos.index, y=conversion_autos.values)
plt.title('Tasa de compra por cantidad de autos')
plt.xlabel('Cantidad de autos')
plt.ylabel('Tasa de compra')


# tasa de compra por grupo en distancia
conversion_distancia = clientes.groupby('Distancia')['ComproBicicleta'].mean()
print(conversion_distancia)
# el 46% de los que viven a menos de 1km compro
# el 36 de los que viven 1-2km
# el 46% de los de 2-5km
# el 32% de los de 5-10km
# el 28 de los de +10km

plt.figure(figsize=(10, 6))
sns.barplot(x=conversion_distancia.index, y=conversion_distancia.values)
plt.title('Tasa de compra por distancia')
plt.xlabel('Distancia')
plt.ylabel('Tasa de compra')














plt.show()






















#       EVALUACION DE LOS DESTINATARIOS

# print(destinatarios.head())


# eliminamos las columnas q no se necesitan

destinatarios.drop("IdCiudad" , axis=1, inplace=True)
destinatarios.drop("Nombre" , axis=1, inplace=True)
destinatarios.drop("Apellido" , axis=1, inplace=True)
destinatarios.drop("Telefono" , axis=1, inplace=True)
destinatarios.drop("Email" , axis=1, inplace=True)
destinatarios.drop("Direccion" , axis=1, inplace=True)
destinatarios.drop("FechaNacimiento" , axis=1, inplace=True)

#usamos una logica parecida a la de clientes para limpiar la columnba de distancia

destinatarios['Distancia'] = destinatarios['Distancia'].str.replace("'", "") # eliminar comillas 
destinatarios['Distancia'] = destinatarios['Distancia'].str.replace(" Km.", "") # eliminar " Km."

destinatarios['Distancia'] = destinatarios['Distancia'].apply(convertir_distancia)


# print(destinatarios.head())

# destinatarios.info()
label_encoders = {}
for column in ['EstadoCivil', 'Genero', 'Educacion', 'Ocupacion', 'Region']:
    le = LabelEncoder()
    destinatarios[column] = le.fit_transform(destinatarios[column])
    label_encoders[column] = le # Guardar los encoders para usar luego en destinatarios




