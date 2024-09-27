using CSV, DataFrames, Random
using LightGBM

#using ZipFile

# Leer el dataset de forma Local
df = CSV.read("G:/Mi unidad/01-Maestria Ciencia de Datos/DMEyF/TPs/dmeyf-2024/datasets/competencia_01_julia.csv", DataFrame)

# Establecer la semilla de aleatoriedad
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
    "seed" => seed
)

### PREPARACIÓN DE LOS DATOS
# Dividir el dataset en entrenamiento y prueba
dataset_train = df[df.foto_mes.<=202104, :]
dataset_test = df[df.foto_mes.==202106, :]

dataset_train = coalesce.(dataset_train, 0)
dataset_test = coalesce.(dataset_test, 0)

# Mapa de etiquetas a valores numéricos
label_map = Dict("BAJA+1" => 0, "BAJA+2" => 0, "CONTINUA" => 1)
label = map(x -> label_map[x], dataset_train.clase_ternaria)

# Convertir los datos a matriz y las etiquetas a vector
data = Matrix(dataset_train[:, Not(:clase_ternaria)])

### ENTRENAMIENTO DEL MODELO
dtrain = LightGBM.DMatrix(data, label=label)
modelo = LightGBM.Booster(params, dtrain)


### PREDICCIÓN
pred = LightGBM.predict(modelo, Matrix(dataset_test[:, Not(:clase_ternaria)]))

probabilidades = 1 .- pred

umbral = 1 / 4

predicciones = probabilidades .> umbral
println("Predicciones de personas que dejan el Banco (BAJA+2): ", sum(predicciones))
dataset_test[!, :Predicted] = Int.(predicciones)


resultado_exportar = select(dataset_test, :numero_de_cliente, :Predicted)

# Exportar resultados a CSV
archivo_numero = "012"

#guardar prediccion en Cloud
CSV.write("/home/joaquintschopp/exp/KA2001/KJulia_" * archivo_numero * ".csv", resultado_exportar)


### CREAR UN ARCHIVO ZIP (backup de los scripts de la versión)
"""
# Archivos a incluir en el ZIP
archivos = [
    "G:/Mi unidad/01-Maestria Ciencia de Datos/DMEyF/TPs/dmeyf-2024/dmeyf2024/00TP-Julia-Tschopp.jl",
    "G:/Mi unidad/01-Maestria Ciencia de Datos/DMEyF/TPs/dmeyf-2024/dmeyf2024/01TP-Julia-Tschopp.jl",
    "G:/Mi unidad/01-Maestria Ciencia de Datos/DMEyF/TPs/dmeyf-2024/dmeyf2024/10TP-Julia-Tschopp.jl",
    "G:/Mi unidad/01-Maestria Ciencia de Datos/DMEyF/TPs/dmeyf-2024/dmeyf2024/15TP-Julia-Tschopp.jl",
    "G:/Mi unidad/01-Maestria Ciencia de Datos/DMEyF/TPs/dmeyf-2024/dmeyf2024/20TP-Julia-Tschopp.jl",
    "G:/Mi unidad/01-Maestria Ciencia de Datos/DMEyF/TPs/dmeyf-2024/dmeyf2024/21TP-Julia-Tschopp.jl"
]

# Ruta del archivo ZIP de destino
ruta_zip = "/home/joaquintschopp/exp/Backup-version-JL/backup_Julia_011.zip"


# Crear el archivo ZIP
ZipFile.zip(ruta_zip, archivos)
"""