using CSV, DataFrames, Random
using LightGBM  # Solo las librerías necesarias


#using ZipFile

# Leer el dataset de forma Local
df = CSV.read("G:/Mi unidad/01-Maestria Ciencia de Datos/DMEyF/TPs/dmeyf-2024/datasets/competencia_01_julia.csv", DataFrame)

seed = 214363
Random.seed!(seed)

### HIPERPARÁMETROS DEL MODELO LIGHTGBM
params = Dict(
    "objective" => "binary",
    "num_iterations" => 250,
    "learning_rate" => 0.57,
    "feature_fraction" => 0.63,
    "min_data_in_leaf" => 31500,
    "num_leaves" => 525,
    "seed" => seed  # Fijar la semilla para garantizar la reproducibilidad
)

### PREPARACIÓN DE LOS DATOS
dataset_train = df[df.foto_mes.<=202104, :]
dataset_test = df[df.foto_mes.==202106, :]

dataset_train = coalesce.(dataset_train, 0)
dataset_test = coalesce.(dataset_test, 0)

#println("Primeros 3 del train:")
#println(typeof(dataset_train))
#println("Primeros 3 del test:")
#println(typeof(dataset_test))

label_map = Dict("BAJA+1" => 0, "BAJA+2" => 0, "CONTINUA" => 1)

# Convertir las etiquetas de la columna clase_ternaria a valores numéricos
label = map(x -> label_map[x], dataset_train.clase_ternaria)
# Convertir los datos a matriz y las etiquetas a vector
data = Matrix(dataset_train[:, Not(:clase_ternaria)])

println("Tipo del Params:")
println(typeof(params))
println("Tipo del data:")
println(typeof(data))
println("Tipo del label:")
println(typeof(label))

println("Primeros 1 de label:")
println(first(label,3))
println("Primeros 3 de data:")
println(first(data,5))

### ENTRENAMIENTO DEL MODELO
# Crear un objeto LGBMEstimator
estimator = LightGBM.LGBMCassifier()

# Entrenar el modelo
modelo = fit!(estimator, data, label, params)

### PREDICCIÓN
pred = LightGBM.predict(modelo, Matrix(dataset_test[:, Not(:clase_ternaria)]))

# Extraer probabilidades de la clase BAJA+2 
probabilidades = 1 .- pred

umbral = 1 / 4

predicciones = probabilidades .> umbral

println("Predicciones de personas que dejan el Banco (BAJA+2): ", sum(predicciones))

# Convertir predicciones booleanas a 1 (BAJA+2) y 0 (CONTINUA/BAJA+1)
dataset_test[!, :Predicted] = Int.(predicciones)

# Seleccionar las columnas a exportar
resultado_exportar = select(dataset_test, :numero_de_cliente, :Predicted)

# Exportar resultados a CSV
archivo_numero = "012"

# guardar prediccion en local
CSV.write("G:/Mi unidad/01-Maestria Ciencia de Datos/DMEyF/TPs/dmeyf-2024/exp/KA2001/KJulia_" * archivo_numero * ".csv", resultado_exportar)

### CREAR UN ARCHIVO ZIP (backup de los scripts de la versión)
"""
archivos = [
    "G:/Mi unidad/01-Maestria Ciencia de Datos/DMEyF/TPs/dmeyf-2024/dmeyf2024/00TP-Julia-Tschopp.jl",
    "G:/Mi unidad/01-Maestria Ciencia de Datos/DMEyF/TPs/dmeyf-2024/dmeyf2024/01TP-Julia-Tschopp.jl",
    "G:/Mi unidad/01-Maestria Ciencia de Datos/DMEyF/TPs/dmeyf-2024/dmeyf2024/10TP-Julia-Tschopp.jl",
    "G:/Mi unidad/01-Maestria Ciencia de Datos/DMEyF/TPs/dmeyf-2024/dmeyf2024/15TP-Julia-Tschopp.jl",
    "G:/Mi unidad/01-Maestria Ciencia de Datos/DMEyF/TPs/dmeyf-2024/dmeyf2024/20TP-Julia-Tschopp.jl",
    "G:/Mi unidad/01-Maestria Ciencia de Datos/DMEyF/TPs/dmeyf-2024/dmeyf2024/21TP-Julia-Tschopp.jl"
]

# Ruta del archivo ZIP de destino
#ruta_zip = "/home/joaquintschopp/exp/Backup-version-JL/backup_Julia_011.zip"
ruta_zip = "G:/Mi unidad/01-Maestria Ciencia de Datos/DMEyF/TPs/dmeyf-2024/exp/Backup-version-JL/backup_Julia_011.zip"

# Crear el archivo ZIP
ZipFile.zip(ruta_zip, archivos)
"""