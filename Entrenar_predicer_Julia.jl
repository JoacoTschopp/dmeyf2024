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


include("Paraetros_Julia_LGBM.jl")

include("./funciones_preproc_julia.jl")


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
    modelo.metric = hiperparametros["metric"]
    modelo.first_metric_only = hiperparametros["first_metric_only"]
    modelo.boost_from_average = hiperparametros["boost_from_average"]
    modelo.feature_pre_filter = hiperparametros["feature_pre_filter"]
    modelo.force_row_wise = hiperparametros["force_row_wise"]
    modelo.verbosity = hiperparametros["verbosity"]
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

    # Escalar los datos
    X_esc = escalar(X)

    # Entrenar el modelo
    fit!(modelo, X_esc, y)
end

# Definir la función de predicción
function predecir(modelo, X)
    X_esc = escalar(X)
    return predict(modelo, X_esc)
end

#########################################################################
#  ACA empieza el programa

# Definir el modelo
modelo = LGBMClassification()

# Definir el dataset
@info "Comienza carga de Dataset"
file = CSV.File("/home/joaquintschopp/buckets/b1/datasets/competencia_julia_ct.csv"; buffer_in_memory=true)
dataset = DataFrame(file)
@info "Fin Carga"

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
predic_data = filter(row -> row.foto_mes == future, dataset)

# Mostrar el tamaño de los datos de predicción
@info "Tamaño de predic_data", size(predic_data)

# Seleccionar `X` y `y` para el entrenamiento
X_train = select(X_train_data, Not(:clase_ternaria)) |> Matrix
y_train = map(x -> x in ["BAJA+1", "BAJA+2"] ? 1 : 0, X_train_data.clase_ternaria)

# Crear `predic`
predic = select(predic_data, Not(:clase_ternaria)) |> Matrix

@info "Entrenamietno del modelo"
entrenar(modelo, X_train, y_train, hiperparametros)

@info "Predicciones sobre el modelo"
# Predecir con el modelo
predicciones = predecir(modelo, predic)


@info "Genero contes y archivos para Kaggle"
generar_csv_cortes(predicciones, numero_de_cliente)
