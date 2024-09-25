# Establecer la variable de entorno para no usar bibliotecas del sistema
#ENV["LIGHTGBM_USE_SYSTEM_LIBS"] = "false"
#ENV["LIGHTGBM_SOURCE"] = "false"


using CSV, DataFrames, Random
using LightGBM  # Solo las librerías necesarias
using ZipFile

# = abspath("~../LightGBM-3.3.5")

# Leer el dataset
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
    "seed" => seed  # Fijar la semilla para garantizar la reproducibilidad
)

### PREPARACIÓN DE LOS DATOS
# Dividir el dataset en entrenamiento y prueba
dataset_train = df[df.foto_mes.<=202104, :]
dataset_test = df[df.foto_mes.==202106, :]

# Reemplazar valores nulos con 0
dataset_train = coalesce.(dataset_train, 0)
dataset_test = coalesce.(dataset_test, 0)

# Mapa de etiquetas a valores numéricos
label_map = Dict("BAJA+1" => 0, "BAJA+2" => 0, "CONTINUA" => 1)

# Convertir las etiquetas de la columna clase_ternaria a valores numéricos
label = map(x -> label_map[x], dataset_train.clase_ternaria)
# Convertir los datos a matriz y las etiquetas a vector
data = Matrix(dataset_train[:, Not(:clase_ternaria)])

### ENTRENAMIENTO DEL MODELO

# Entrenar el modelo
modelo = LightGBM.fit!(params, data, label)

### PREDICCIÓN
# Aplicar el modelo al dataset de prueba
pred = LightGBM.predict(modelo, Matrix(dataset_test[:, Not(:clase_ternaria)]))

# Extraer probabilidades de la clase BAJA+2 (asumiendo que la clase BAJA+2 es la segunda clase)
probabilidades = 1 .- pred  # Columna de probabilidades para la clase "BAJA+2"

# Definir el umbral para la clase "BAJA+2"
umbral = 1 / 4

# Realizar predicciones según el umbral definido
predicciones = probabilidades .> umbral

# Imprimir cuántas personas se predicen que dejan el Banco (BAJA+2)
println("Predicciones de personas que dejan el Banco (BAJA+2): ", sum(predicciones))

# Convertir predicciones booleanas a 1 (BAJA+2) y 0 (CONTINUA/BAJA+1)
dataset_test[!, :Predicted] = Int.(predicciones)

# Seleccionar las columnas a exportar
resultado_exportar = select(dataset_test, :numero_de_cliente, :Predicted)

# Exportar resultados a CSV
archivo_numero = "012"
CSV.write("G:/Mi unidad/01-Maestria Ciencia de Datos/DMEyF/TPs/dmeyf-2024/exp/KA2001/KJulia_" * archivo_numero * ".csv", resultado_exportar)

### CREAR UN ARCHIVO ZIP (backup de los scripts de la versión)

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
ruta_zip = "G:/Mi unidad/01-Maestria Ciencia de Datos/DMEyF/TPs/dmeyf-2024/dmeyf2024/Backup-version-JL/backup_Julia_011.zip"

# Crear el archivo ZIP
ZipFile.zip(ruta_zip, archivos...)