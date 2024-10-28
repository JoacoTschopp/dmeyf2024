using Pkg
#Pkg.add("MLJ")
#Pkg.add("LightGBM")
#Pkg.add("DataFrames")
#Pkg.add("Random")
#Pkg.add("CSV")



using LightGBM
using DataFrames
using Statistics
using CSV, DataFrames
using Dagger
using Dates

include("Paraetros_Julia_LGBM.jl")

include("./Funciones_julia.jl")


# Definir la función de escalado
function escalar(X)
    mean_X = mean(X, dims=1)
    std_X = std(X, dims=1)
    return (X .- mean_X) ./ std_X
end

# Definir la función de entrenamiento
function entrenar(modelo, X, y, hiperparametros)
    # Cargar los hiperparámetros en el modelo
    modelo.boosting = hiperparametros["boosting"]
    modelo.objective = hiperparametros["objective"]
    modelo.metric = [hiperparametros["metric"]]
    #modelo.first_metric_only = hiperparametros["first_metric_only"]
    modelo.boost_from_average = hiperparametros["boost_from_average"]
    modelo.feature_pre_filter = hiperparametros["feature_pre_filter"]
    modelo.force_row_wise = hiperparametros["force_row_wise"]
    #modelo.verbosity = hiperparametros["verbosity"]
    modelo.max_depth = hiperparametros["max_depth"]
    modelo.min_gain_to_split = hiperparametros["min_gain_to_split"]
    modelo.min_sum_hessian_in_leaf = hiperparametros["min_sum_hessian_in_leaf"]
    modelo.lambda_l1 = hiperparametros["lambda_l1"]
    modelo.lambda_l2 = hiperparametros["lambda_l2"]
    modelo.max_bin = hiperparametros["max_bin"]
    modelo.num_iterations = hiperparametros["num_iterations"]
    modelo.bagging_fraction = hiperparametros["bagging_fraction"]
    modelo.pos_bagging_fraction = hiperparametros["pos_bagging_fraction"]
    modelo.neg_bagging_fraction = hiperparametros["neg_bagging_fraction"]
    modelo.is_unbalance = hiperparametros["is_unbalance"]
    modelo.scale_pos_weight = hiperparametros["scale_pos_weight"]
    modelo.drop_rate = hiperparametros["drop_rate"]
    modelo.max_drop = hiperparametros["max_drop"]
    modelo.skip_drop = hiperparametros["skip_drop"]
    modelo.extra_trees = hiperparametros["extra_trees"]
    modelo.learning_rate = hiperparametros["learning_rate"]
    modelo.feature_fraction = hiperparametros["feature_fraction"]
    modelo.num_leaves = hiperparametros["num_leaves"]
    modelo.min_data_in_leaf = hiperparametros["min_data_in_leaf"]
    modelo.num_class = 1

    # Convertir valores faltantes en X a 0
    X = replace(X, missing => 0.0)  # Cambia a 0.0 para asegurarte de que sea del tipo correcto

    # Contar los valores faltantes
    num_missing = count(ismissing.(X))

    @info "Número total de valores faltantes en X_train: ", num_missing

    # Convertir valores faltantes en y a 0
    y = replace(y, missing => 0)  # Cambia a 0 para que y sea un vector de enteros o booleanos

    # Asegúrate de que `y` y `X` sean compatibles
    if size(X, 1) != length(y)
        throw(ArgumentError("Las dimensiones de X y y no coinciden después de reemplazar valores faltantes."))
    end

    y = Vector(y)

    # Entrenar el modelo
    try
        @info "Entrenando el modelo..."
        fit!(modelo, X, y, verbosity = -1)
    catch e
        println("Error durante el entrenamiento: ", e)
    end
end

# Definir la función de predicción
function predecir(modelo, X)
    X = replace(X, missing => 0.0)
    return predict(modelo, X)
end

#########################################################################
#  ACA empieza el programa

# Definir el modelo
modelo = LGBMClassification()

# Definir el dataset
@info "Comienza carga de Dataset - $(now())"
file = CSV.File("/home/joaquintschopp/buckets/b1/datasets/competencia_julia_ct.csv"; buffer_in_memory=true)
dataset = DataFrame(file)
@info "Fin Carga - $(now())"

@info "Pipline de entrenamiento"
hiperparametros = param_local["lgb_param"]

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
y_train = map(x -> x in ["BAJA+1", "BAJA+2"] ? 1 : 0, X_train_data.clase_ternaria)

# Imprimir los valores únicos de y_train
println("Valores únicos de y_train:")
println(unique(y_train))
println(typeof(y_train))
y_train = collect(Int64[y_train...])
println("El tipo de y_train es: ", typeof(y_train))
# Crear `predic`

println("Dimensiones de X_train: ", size(X_train))
println("Dimensiones de y_train: ", size(y_train))

predic = select(predic_data, Not(:clase_ternaria)) |> Matrix

@info "Entrenamietno del modelo"
entrenar(modelo, X_train, y_train, hiperparametros)

@info "Predicciones sobre el modelo"
# Predecir con el modelo
predicciones = predecir(modelo, predic)


@info "Genero contes y archivos para Kaggle"
generar_csv_cortes(predicciones, numero_de_cliente)
