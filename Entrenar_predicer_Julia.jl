using Pkg
#Pkg.add("MLJ")
#Pkg.add("LightGBM")
#Pkg.add("DataFrames")
#Pkg.add("Random")
Pkg.add("CSV")

using LightGBM
using DataFrames
using Statistics
using CSV, DataFrames

# Definir las variables importantes
param_local = Dict(
    "future" => [202108],
    "final_train" => Dict(
        "undersampling" => 1.0,
        "clase_minoritaria" => ["BAJA+1", "BAJA+2"],
        "training" => [202106, 202105, 202104, 202103, 202102, 202101, 
                        202005, 202006, 202007, 202008, 202009, 202010, 202011, 202012]
    ),
    "train" => Dict(
        "training" => [202104, 202103, 202102, 202101, 
                          202012, 202011, 202005, 202006, 202007, 202008, 202009, 202010],
        "validation" => [202105],
        "testing" => [202106],
        "undersampling" => 0.75,
        "clase_minoritaria" => ["BAJA+1", "BAJA+2"]
    ),
    "lgb_param" => Dict(
        "boosting" => "gbdt",
        "objective" => "binary",
        "metric" => "custom",
        "first_metric_only" => true,
        "boost_from_average" => true,
        "feature_pre_filter" => false,
        "force_row_wise" => true,
        "verbosity" => -100,
        "max_depth" => -1,
        "min_gain_to_split" => 0.0,
        "min_sum_hessian_in_leaf" => 0.001,
        "lambda_l1" => 0.0,
        "lambda_l2" => 0.0,
        "max_bin" => 31,
        "num_iterations" => 9999,
        "bagging_fraction" => 1.0,
        "pos_bagging_fraction" => 1.0,
        "neg_bagging_fraction" => 1.0,
        "is_unbalance" => false,
        "scale_pos_weight" => 1.0,
        "drop_rate" => 0.1,
        "max_drop" => 50,
        "skip_drop" => 0.5,
        "extra_trees" => false,
        "learning_rate" => [0.02],#, 0.3],
        "feature_fraction" => [0.5],# 0.9],
        "num_leaves" => [2048], #8,
        "min_data_in_leaf" => [100]#, 10000]
    )
)

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

function generar_csv_cortes(predicciones::DataFrame)
    # Verificar que el DataFrame `predicciones` tenga las columnas requeridas
    if !all(["numero_de_cliente", "Predicted"] .∈ names(predicciones))
        error("El DataFrame debe contener las columnas `numero_de_cliente` y `Predicted`.")
    end
    
    # Ordenar el DataFrame de mayor a menor según la columna `Predicted`
    sort!(predicciones, :Predicted, rev=true)

    # Definir los cortes de 8500 a 13500, sumando de a 500
    cortes = 8500:500:13500

    # Generar CSV para cada corte
    for corte in cortes
        # Crear copia de `predicciones` y asignar 1 para los primeros `corte` y 0 para el resto
        resultados_corte = copy(predicciones)
        resultados_corte.Predicted .= 0
        resultados_corte[1:corte, :Predicted] .= 1

        # Guardar el archivo CSV con el nombre correspondiente al corte
        nombre_archivo = "~/buckets/b1/exportaJulia/predicciones_corte_$corte.csv"
        CSV.write(nombre_archivo, resultados_corte; header=["numero_de_cliente", "Predicted"])
        println("Archivo generado: $nombre_archivo")
    end
end


#########################################################################
#  ACA empieza el programa

# Definir el modelo
modelo = LGBMClassification()

# Definir el dataset
dataset = CSV.read("~/buckets/b1/datasets/competencia_02_ct.csv.gz")

hiperparametros = param_local["lgb_param"]

# Definir `future` y `training` utilizando los valores del diccionario
future = param_local["future"]
training = param_local["final_train"]["training"]

# Filtrar los datos para `X_train` y `predic`
X_train_data = dataset[dataset[!, :foto_mes] .∈ training, :]
predic_data = dataset[dataset[!, :foto_mes] .== future[1], :]

# Seleccionar `X` y `y` para el entrenamiento
X_train = select(X_train_data, Not(:clase_ternaria)) |> Matrix
y_train = map(x -> x in ["BAJA+1", "BAJA+2"] ? 1 : 0, X_train_data.clase_ternaria)

# Crear `predic`
predic = select(predic_data, Not(:clase_ternaria)) |> Matrix


entrenar(modelo, X_train, y_train, hiperparametros)

# Predecir con el modelo
predicciones = predecir(modelo, predic)

generar_csv_cortes(predicciones, numero_de_cliente)
