using DataFrames
using Random
using Statistics

# Configuración de la corrida
PARAM = Dict(
    "experimento" => "HT4740Julia",
    "semilla_primigenia" => 214363,
    "dataset" => "G:/Mi unidad/01-Maestria Ciencia de Datos/DMEyF/TPs/dmeyf-2024/datasets/competencia_01_julia.csv",
    "input" => Dict(
        "training" => [202104]
    ),
    "hyperparametertuning" => Dict(
        "iteraciones" => 100,
        "xval_folds" => 5,
        "POS_ganancia" => 273000,
        "NEG_ganancia" => -7000
    )
)

# Definición de los hiperparámetros y sus rangos
hs = Dict(
    "num.trees" => (50, 200),
    "max.depth" => (5, 20),
    "min.node.size" => (10, 1000),
    "mtry" => (5, 20),
    "learning.rate" => (0.01, 0.1)
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
function ranger_Simple(fold_test, pdata, param)
    modelo = RandomForestClassifier(
        n_trees = param["num.trees"],
        max_depth = param["max.depth"],
        min_samples_split = param["min.node.size"],
        mtry = param["mtry"]
    )
    modelo = fit!(modelo, pdata[fold != fold_test])
    prediccion = predict(modelo, pdata[fold == fold_test])
    ganancia_testing = sum((prediccion .> 1 / 40) .* ifelse(pdata[fold == fold_test, :clase_binaria] .== "POS", PARAM["hyperparametertuning"]["POS_ganancia"], PARAM["hyperparametertuning"]["NEG_ganancia"]))
    return ganancia_testing
end

# Función para realizar la validación cruzada
function ranger_CrossValidation(data, param, qfolds, pagrupa, semilla)
    divi = repeat([1], qfolds)
    data = particionar(data, divi, pagrupa, seed = semilla)
    ganancias = [ranger_Simple(i, data, param) for i in 1:qfolds]
    data[!, :fold] = nothing
    ganancia_promedio = mean(ganancias)
    ganancia_promedio_normalizada = ganancia_promedio * qfolds
    return ganancia_promedio_normalizada
end

# Función para estimar la ganancia
function EstimarGanancia_ranger(x)
    GLOBAL_iteracion += 1
    xval_folds = PARAM["hyperparametertuning"]["xval_folds"]
    ganancia = ranger_CrossValidation(dataset, x, xval_folds, "clase_binaria", PARAM["semillas"][1])
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
    funcion_optimizar = EstimarGanancia_ranger
    # Configuración de la búsqueda bayesiana
    obj.fun = funcion_optimizar
    # Configuración del control de la búsqueda
    ctrl = Dict(
        "iteraciones" => PARAM["hyperparametertuning"]["iteraciones"],
        "save.on.disk.at.time" => 600,
        "save.file.path" => "kbayesiana"
    )
    return obj.fun, ctrl
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
    obj.fun, ctrl = configureMlr()
    # Inicio de la optimización bayesiana
    if !isfile("kbayesiana")
        run = [EstimarGanancia_ranger(Dict("num.trees" => 100, "max.depth" => 10, "min.node.size" => 10, "mtry" => 5, "learning.rate" => 0.1)) for _ in 1:ctrl["iteraciones"]]
    else
        run = [EstimarGanancia_ranger(Dict("num.trees" => 100, "max.depth" => 10, "min.node.size" => 10, "mtry" => 5, "learning.rate" => 0.1)) for _ in 1:ctrl["iteraciones"]]
    end
    return run
end

# Ejecución del script
main()