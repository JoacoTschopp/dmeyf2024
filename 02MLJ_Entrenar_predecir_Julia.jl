using Pkg
#Pkg.add("MLJ")
#Pkg.add("LightGBM")
#Pkg.add("DataFrames")
#Pkg.add("Random")
#Pkg.add("CSV")
#Pkg.add("YMAL")
#Pkg.add("Dates")
Pkg.add(["MLJ", "LightGBM", "MLJTuning", "DataFrames", "MLJPipeline"])



using LightGBM
using DataFrames
using Statistics
using CSV, DataFrames
using Dates
using YAML
using Random
using MLJ
using MLJTuning


param_local = YAML.load_file("Parametros_Julia_LGBM.yaml")

include("./Funciones_julia.jl")

#########################################################################

#3. Crear el Pipeline y Configurar LightGBMClassification

# Definir el modelo de LightGBM
modelo = @load LightGBMClassifier pkg=LightGBM

# Crear un pipeline (por ejemplo, con escalado de características antes del modelo)
scaler = Standardizer()                # Paso opcional de escalado
pipeline = scaler |> modelo             # Definir pipeline

#4. Definir el Rango de Hiperparámetros para la Optimización Bayesiana
# Definir los rangos de hiperparámetros
range = [
    range(modelo, :learning_rate, lower=0.01, upper=0.3),
    range(modelo, :num_leaves, lower=20, upper=100),
    range(modelo, :max_depth, lower=3, upper=10)
]

#5. Configurar la Optimización Bayesiana y Entrenar el Pipeline
# Configurar la búsqueda de hiperparámetros con optimización bayesiana
tuned_pipeline = TunedModel(
    model=pipeline,
    tuning=BayesOpt(),                  # Optimización bayesiana
    resampling=CV(nfolds=5),            # Validación cruzada
    range=range,
    measure=LogLoss())                  # Métrica de evaluación

# Dividir los datos en entrenamiento y prueba
X, y = @load_iris              # O tu propio dataset
X_train, X_test, y_train, y_test = partition(X, y, 0.7, rng=123)

# Entrenar el pipeline ajustado
mach_tuned = machine(tuned_pipeline, X_train, y_train)
fit!(mach_tuned)

#6. Evaluar el Modelo
# Hacer predicciones y evaluar
y_pred = predict(mach_tuned, X_test)
accuracy = mean(y_pred .== y_test)
println("Accuracy: ", accuracy)

#7. Guardar y Cargar el Modelo Entrenado

using Serialization

# Guardar el modelo
Serialization.serialize("modelo_lightgbm.jls", mach_tuned)

# Cargar el modelo
mach_tuned_cargado = Serialization.deserialize("modelo_lightgbm.jls")

# Usar el modelo cargado para predecir
y_pred_cargado = predict(mach_tuned_cargado, X_test)




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
#entrenar(modelo, X_train, y_train)
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
