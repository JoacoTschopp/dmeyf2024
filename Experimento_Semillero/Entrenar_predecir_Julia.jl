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

@info "Pipline de entrenamiento"

# Definir `future` y `training` utilizando los valores del diccionario
future = param_local["future"]
training = param_local["final_train"]["training"]

# Filtrar los datos para `X_train` y `predic`
X_train_data = filter(row -> row.foto_mes in training, dataset)
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
#LightGBM.savemodel(modelo, "D:/DmEyF_Julia/exportaJulia/modelo_entrenado.txt")

# Cargar el modelo en otra sesión o script
#@info "Cargo modelo guardado"
#LightGBM.loadmodel!(modelo, "D:/DmEyF_Julia/exportaJulia/modelo_entrenado.txt")


@info "Predicciones sobre el modelo"
# Predecir con el modelo
predicciones = predecir(modelo, X_future)

# Extraer el número de cliente de `predic_data`
numero_de_cliente = predic_data[:, :numero_de_cliente]  
# Crear DataFrame con `numero_de_cliente` y las `predicciones`
df_predicciones = DataFrame(numero_de_cliente=numero_de_cliente, Predicted=vec(predicciones))
    

@info "Genero contes y archivos para Kaggle"
generar_csv_cortes(df_predicciones)
