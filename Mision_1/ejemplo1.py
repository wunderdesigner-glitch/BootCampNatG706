import pandas as pd
datos={
    "Nombre": ["Ana", "Luis", "Pedro"],
    "Edad": [28, 34, 22],
    "nota": [3.5, 4.2, 2.8],
    "ciudad" : ["Bogotá", "Pereira", "Medellín"]
}
df = pd.DataFrame(datos)
print(df) 

#Promedio de notas
print("Promedio de notas:", df["nota"].mean()) 

#Promedio de notas
print("Promedio de edades:", df["Edad"].mean())
