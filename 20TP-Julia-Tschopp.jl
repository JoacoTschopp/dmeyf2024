using CategoricalArrays
using CSV, DataFrames, Random, Statistics
using Primes
using DecisionTree, Impute
using Base.Threads
using Printf
using ZipFile

df = CSV.read("G:/Mi unidad/01-Maestria Ciencia de Datos/DMEyF/TPs/dmeyf-2024/datasets/competencia_01_julia.csv", DataFrame)

### ARMADO DE ARBOL E HYPERPARAMETROS

struct ttree
    n_subfeatures::UInt
    maxdepth::UInt
    min_samples_split::UInt
    min_samples_leaf::UInt
    min_purity_increase::Float64
 end
 
#ptree = ttree(0, 5, 800, 10, -1.0)

#ptree = ttree(0, 7, 800, 10, -1.0)

ptree = ttree(1, 7, 400, 30, -1.0)

####ENTRENAMIENTO Y CALCULO DE MEDIAS

# Filtrar dataset para el entrenamiento y testeo
dataset_train = df[df.foto_mes .<= 202104, :]
dataset_test = df[df.foto_mes .== 202106, :]

# Imputar valores nulos en el dataset de entrenamiento
dataset_train = Impute.substitute(dataset_train)
dataset_test = Impute.substitute(dataset_test)
#Imputar por 0
dataset_train = coalesce.(dataset_train, 0)
dataset_test = coalesce.(dataset_test, 0)

# formato para  DecisionTrees
dataset_clase = string.(dataset_train[ :, :clase_ternaria ])
dataset_matriz = Matrix( dataset_train[ :, Not(:clase_ternaria)] )

# Entrenamiento del modelo
modelo = DecisionTree.build_tree(
    dataset_clase,
    dataset_matriz,
    ptree.n_subfeatures,
    ptree.maxdepth,
    ptree.min_samples_leaf,
    ptree.min_samples_split,
    ptree.min_purity_increase
)

# Aplicar el modelo al dataset de prueba
pred = apply_tree_proba(
    modelo, 
    Matrix(dataset_test[:, Not(:clase_ternaria)]),
    ["BAJA+1", "BAJA+2", "CONTINUA"]
)

# Extraer probabilidades de la clase positiva (BAJA+2)
probabilidades_positivas = pred[:, 2]  # Probabilidad de "BAJA+2"

# Establecer el umbral y realizar predicciones
umbral = 1 / 40
predicciones_umbral = probabilidades_positivas .> umbral

println("Predicciones de personas que dejan el Banco: ", sum(predicciones_umbral))

# Convertir predicciones booleanas a 1 y 0
dataset_test[!, :Predicted] = Int.(predicciones_umbral)

# Seleccionar columnas a exportar
resultado_exportar = select(dataset_test, :numero_de_cliente, :Predicted)

# Exportar resultados a CSV
archivo_numero = "011"
CSV.write("G:/Mi unidad/01-Maestria Ciencia de Datos/DMEyF/TPs/dmeyf-2024/exp/KA2001/KJulia_" * archivo_numero * ".csv", resultado_exportar)

###Guardar archivo zip con copia segun version del soft que genero la prediccion...
# Archivos a incluir en el ZIP
archivos = [
    "G:/Mi unidad/01-Maestria Ciencia de Datos/DMEyF/TPs/dmeyf-2024/dmeyf2024/00TP-Julia-Tschopp.jl",
    "G:/Mi unidad/01-Maestria Ciencia de Datos/DMEyF/TPs/dmeyf-2024/dmeyf2024/01TP-Julia-Tschopp.jl",
    "G:/Mi unidad/01-Maestria Ciencia de Datos/DMEyF/TPs/dmeyf-2024/dmeyf2024/10TP-Julia-Tschopp.jl",
    "G:/Mi unidad/01-Maestria Ciencia de Datos/DMEyF/TPs/dmeyf-2024/dmeyf2024/15TP-Julia-Tschopp.jl",
    "G:/Mi unidad/01-Maestria Ciencia de Datos/DMEyF/TPs/dmeyf-2024/dmeyf2024/20TP-Julia-Tschopp.jl"
]

# Ruta del archivo ZIP de destino
ruta_zip = "G:/Mi unidad/01-Maestria Ciencia de Datos/DMEyF/TPs/dmeyf-2024/dmeyf2024/Backup-version-JL/backup_Julia_" * archivo_numero * ".zip"

# Crear un archivo ZIP
ZipFile.zip(ruta_zip, archivos...)