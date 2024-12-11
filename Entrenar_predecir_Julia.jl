using Pkg
#Pkg.add("MLJ")
#Pkg.add("LightGBM")
#Pkg.add("DataFrames")
#Pkg.add("Random")
#Pkg.add("CSV")
#Pkg.add("YMAL")
#Pkg.add("Dates")

using LightGBM
using DataFrames
using Statistics
using CSV, DataFrames
using Dates
using YAML
using Random
using MLJ

param_local = YAML.load_file("Parametros_Julia_LGBM.yaml")

include("./Funciones_julia.jl")



#########################################################################
#  ACA empieza el programa
#Defino semilla
Random.seed!(param_local["semilla_primigenia"])

# Definir el modelo
modelo = LGBMClassification()

# Definir el dataset
@info "Comienza carga de Dataset - $(now())"
file = CSV.File(param_local["dataset"]; buffer_in_memory=true)
dataset = DataFrame(file)
@info "Fin Carga - $(now())"
############################################################################
#Generamos clase clase_ternaria
@info "Comienza clase_ternaria - $(now())"
dataset = gen_calse_ternaria(dataset)

@info "Fin clase_ternaria"
############################################################################
@info "Comienza BO - $(now())"
println("Filtrando y submuestreando el dataset...")

# Crear una copia del dataset para trabajar con `train_bo`
dataset_bo = deepcopy(dataset)

# Extraer parámetros de train_bo
train_bo_params = param_local["train_bo"]
training_bo = train_bo_params["training"]
validation_bo = train_bo_params["validation"]
testing_bo = train_bo_params["testing"]
undersampling_bo = train_bo_params["undersampling"]
clase_minoritaria = train_bo_params["clase_minoritaria"]

# Filtrar los conjuntos de validación y testing
validation_testing_bo = filter(row -> row.foto_mes in validation_bo || row.foto_mes in testing_bo, dataset_bo)

# Filtrar el conjunto de entrenamiento y aplicar submuestreo
training_bo_dataset = filter(row -> row.foto_mes in training_bo, dataset_bo)

# Separar clases mayoritaria y minoritaria
majority_class_rows = collect(eachrow(filter(row -> row.clase_ternaria != clase_minoritaria, training_bo_dataset)))
minority_class_rows = filter(row -> row.clase_ternaria == clase_minoritaria, training_bo_dataset)

# Submuestrear aleatoriamente la clase mayoritaria
sample_size_bo = min(undersampling_bo, length(majority_class_rows))
selected_majority = DataFrame(shuffle(majority_class_rows)[1:sample_size_bo])

# Combinar todo el dataset en una sola llamada
dataset_bo = vcat(minority_class_rows, selected_majority, validation_testing_bo, cols=:union)

# Guardar el DataFrame combinado en un archivo CSV
output_file_path = joinpath(param_local["experimento"], "dataset_bo.csv")
CSV.write(output_file_path, dataset_bo)

println("Proceso completado. El archivo se ha guardado en: $output_file_path")

HT_BO_Julia(dataset_bo, validation_bo, testing_bo, train_bo_params, param_local)

@info "Fin BO - $(now())"
############################################################################

@info "Pipline de entrenamiento FINAL"

######################################
#
# Preaprado de dataset para entrenamiento FINAL

# Definir `future` y `training` utilizando los valores del diccionario
future = param_local["future"]
training = param_local["final_train"]["training"]
undersampling_final_training = param_local["final_train"]["undersampling"]

# Filtrar los datos para `X_train` y `predic`
X_train_data = filter(row -> row.foto_mes in training, dataset)

# Separar las filas de la clase CONTINUA
continua_rows = filter(row -> row.clase_ternaria == "CONTINUA", X_train_data)

# Filtrar aleatoriamente las filas de CONTINUA según undersampling_final_training
sample_size = min(undersampling_final_training, size(continua_rows, 1))  # Asegurar que no se exceda el tamaño real
selected_continua = shuffle(continua_rows)[1:sample_size]

# Combinar las filas seleccionadas de CONTINUA con las otras clases
other_rows = filter(row -> row.clase_ternaria != "CONTINUA", X_train_data)
X_train_data = vcat(other_rows, selected_continua)

println(size(X_train_data))  # Tamaño del DataFrame filtrado

# Filtrado para `predic_data`
@info "Iniciando filtrado de datos para predicción"
print(future)
predic_data = filter(row -> row.foto_mes in future, dataset)

# Mostrar el tamaño de los datos de predicción
@info "Tamaño de predic_data", size(predic_data)

# Seleccionar `X` y `y` para el entrenamiento
X_train = select(X_train_data, Not(:clase_ternaria)) |> Matrix
y_train = map(x -> x in param_local["final_train"]["clase_minoritaria"] ? 1 : 0, X_train_data.clase_ternaria)

# Imprimir los valores únicos de y_train Varificacion de Vector
#println("Valores únicos de y_train:")
#println(unique(y_train))
#println(typeof(y_train))

y_train = collect(Int64[y_train...])
println("El tipo de y_train es: ", typeof(y_train))

# Crear `predic`
X_future = select(predic_data, Not(:clase_ternaria)) |> Matrix

##Con la intencion de Generar solo una vez el modelo y poder jugar con el modelo guardado se agregan estas lineas luego de entrenado
@info "Entrenamietno del modelo"
entrenar(modelo, X_train, y_train)
@info "FIN Entrenamietno del modelo - $(now())"

# Guardar el modelo en un archivo
@info "Guardo el modelo"
LightGBM.savemodel(modelo, joinpath(param_local["experimento"], "modelo_entrenado.txt"))

# Cargar el modelo en otra sesión o script
#@info "Cargo modelo guardado"
#LightGBM.loadmodel!(modelo, "D:/DmEyF_Julia/exportaJulia/modelo_entrenado.txt")


######################################################################################
#
# Generamos prediccion y cortes.

@info "Predicciones sobre el modelo"
# Predecir con el modelo
predicciones = predecir(modelo, X_future)

# Extraer el número de cliente de `predic_data`
numero_de_cliente = predic_data[:, :numero_de_cliente]
# Crear DataFrame con `numero_de_cliente` y las `predicciones`
df_predicciones = DataFrame(numero_de_cliente=numero_de_cliente, Predicted=vec(predicciones))


@info "Genero contes y archivos para Kaggle"
generar_csv_cortes(df_predicciones)
