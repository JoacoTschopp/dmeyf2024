using DataFrames
using LightGBM
using Random
using Statistics

# Configuración de la corrida
PARAM = Dict(
    "experimento" => "HT4220MisHiper",
    "semilla_primigenia" => 214363,
    "dataset" => "G:/Mi unidad/01-Maestria Ciencia de Datos/DMEyF/TPs/dmeyf-2024/datasets/competencia_01_julia.csv",
    "input" => Dict(
        "training" => [202104]
    ),
    "hyperparametertuning" => Dict(
        "iteraciones" => 150,
        "xval_folds" => 5,
        "POS_ganancia" => 273000,
        "NEG_ganancia" => -7000
    )
)

# Definición de los hiperparámetros y sus rangos
hs = Dict(
    "learning_rate" => (0.01, 0.3),
    "num_leaves" => (8, 1024),
    "feature_fraction" => (0.1, 1.0),
    "min_data_in_leaf" => (1, 8000),
    "envios" => (5000, 15000),
    "max_depth" => (5, 15),
    "min_gain_to_split" => (0.1, 0.5),
    "lambda_l1" => (0.01, 0.1),
    "lambda_l2" => (0.01, 0.1),
    "bagging_fraction" => (0.5, 0.9),
    "bagging_freq" => (1, 5)
)

# Función para particionar el dataset
function particionar(data, division, agrupa, campo = "fold", start = 1, seed = nothing)
    if !isnothing(seed)
        Random.seed!(seed)
    end
    bloque = repeat(division, inner = ceil(Int, size(data, 1) / length(division)))
    data[!, campo] = sample(bloque, size(data, 1))
    return data
end

# Función para entrenar el modelo
function lightgbm_Simple(fold_test, pdata, param)
    modelo = LightGBMClassifier(
        learning_rate = param["learning_rate"],
        num_leaves = param["num_leaves"],
        feature_fraction = param["feature_fraction"],
        min_data_in_leaf = param["min_data_in_leaf"],
        envios = param["envios"],
        max_depth = param["max_depth"],
        min_gain_to_split = param["min_gain_to_split"],
        lambda_l1 = param["lambda_l1"],
        lambda_l2 = param["lambda_l2"],
        bagging_fraction = param["bagging_fraction"],
        bagging_freq = param["bagging_freq"]
    )
    modelo = fit!(modelo, pdata[fold != fold_test])
    prediccion = predict(modelo, pdata[fold == fold_test])
    ganancia_testing = sum((prediccion .> 1 / 40) .* ifelse(pdata[fold == fold_test, :clase_binaria] .== "POS", PARAM["hyperparametertuning"]["POS_ganancia"], PARAM["hyperparametertuning"]["NEG_ganancia"]))
    return ganancia_testing
end

# Función para realizar la validación cruzada
function lightgbm_CrossValidation(data, param, qfolds, pagrupa, semilla)
    divi = repeat([1], qfolds)
    data = particionar(data, divi, pagrupa, seed = semilla)
    ganancias = [lightgbm_Simple(i, data, param) for i in 1:qfolds]
    data[!, :fold] = nothing
    ganancia_promedio = mean(ganancias)
    ganancia_promedio_normalizada = ganancia_promedio * qfolds
    return ganancia_promedio_normalizada
end

# Función para estimar la ganancia
function EstimarGanancia_lightgbm(x)
    GLOBAL_iteracion += 1
    xval_folds = PARAM["hyperparametertuning"]["xval_folds"]
    ganancia = lightgbm_CrossValidation(dataset, x, xval_folds, "clase_binaria", PARAM["semillas"][1])
    # Logueo
    xx = Dict(x)
    xx["xval_folds"] = xval_folds
    xx["ganancia"] = ganancia
    xx["iteracion"] = GLOBAL_iteracion
    println(xx)
    # Si es ganancia superadora la almaceno en mejor
    if ganancia > GLOBAL_mejor
        GLOBAL_mejor = ganancia
        println(xx)
    end
    return ganancia
end

# Configuración de la optimización bayesiana
function configureMlr()
    # Configuración de la función de optimización
    funcion_optimizar = EstimarGanancia_lightgbm
    # Configuración del control de la búsqueda
    ctrl = Dict(
        "iteraciones" => PARAM["hyperparametertuning"]["iteraciones"],
        "save.on.disk.at.time" => 600,
        "save.file.path" => "kbayesiana"
    )
    return funcion_optimizar, ctrl
end

# Inicio de la optimización bayesiana
function main()
    # Configuración de la corrida
    PARAM["semillas"] = [214363, 214364]
    # Carga del dataset
    dataset = DataFrame(CSV.File(PARAM["dataset"]))
    # Particionamiento del dataset
    dataset = particionar(dataset, [1, 1, 1, 1, 1], "clase_binaria", seed = PARAM["semillas"][1])
    # Configuración de la optimización bayesiana
    funcion_optimizar, ctrl = configureMlr()
    # Inicio de la optimización bayesiana
    if !isfile("kbayesiana")
        run = [EstimarGanancia_lightgbm(Dict("learning_rate" => 0.1, "num_leaves" => 100, "feature_fraction" => 0.5, "min_data_in_leaf" => 10, "envios" => 5000, "max_depth" => 10, "min_gain_to_split" => 0.2, "lambda_l1" => 0.05, "lambda_l2" => 0.05, "bagging_fraction" => 0.7, "bagging_freq" => 2)) for _ in 1:ctrl["iteraciones"]]
    else
        run = [EstimarGanancia_lightgbm(Dict("learning_rate" => 0.1, "num_leaves" => 100, "feature_fraction" => 0.5, "min_data_in_leaf" => 10, "envios" => 5000, "max_depth" => 10, "min_gain_to_split" => 0.2, "lambda_l1" => 0.05, "lambda_l2" => 0.05, "bagging_fraction" => 0.7, "bagging_freq" => 2)) for _ in 1:ctrl["iteraciones"]]
    end
    return run
end

# Ejecución del script
main()