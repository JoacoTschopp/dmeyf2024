using LightGBM
using CSV
using DataFrames
using MLJ

#####################################################################################
#
#Genera Clase clase_ternaria

function gen_calse_ternaria(df::DataFrame)
    periodo_anterior(x::Integer) = x % 100 > 1 ? x - 1 : 12 + (div(x, 100) - 1) * 100

    size(df)

    sort!(df, [:numero_de_cliente, :foto_mes])

    global periodo_ultimo = maximum(df.foto_mes)
    global periodo_anteultimo = periodo_anterior(periodo_ultimo)

    # assign most common class values
    df.clase_ternaria = @. ifelse(df.foto_mes < periodo_anteultimo, "CONTINUA", missing)

    # pre compute sequential time
    periodo = @. div(df.foto_mes, 100) * 12 + df.foto_mes % 100

    global last = nrow(df)

    for i in 1:last
        if df.foto_mes[i] <= periodo_anteultimo && i < last &&
           (df.numero_de_cliente[i] != df.numero_de_cliente[i+1] || df.numero_de_cliente[i] == df.numero_de_cliente[i+1] && periodo[i+1] > periodo[i] + 1)
            df.clase_ternaria[i] = "BAJA+1"
        end

        if df.foto_mes[i] < periodo_anteultimo && i + 1 < last && df.numero_de_cliente[i] == df.numero_de_cliente[i+1] && periodo[i+1] == periodo[i] + 1 &&
           (df.numero_de_cliente[i+1] != df.numero_de_cliente[i+2] || df.numero_de_cliente[i+1] == df.numero_de_cliente[i+2] && periodo[i+2] > periodo[i+1] + 2)
            df.clase_ternaria[i] = "BAJA+2"
        end
    end

    return df
end

#####################################################################################
#
# Funcion apra buscar Hiperparametros con una bo

# Función para entrenamiento y optimización
function HT_BO_Julia(dataset_bo::DataFrame, validation_data_bo::Vector, testing_data_bo::Vector, train_bo_params::Dict, parametros::Dict)
    println("Iniciando entrenamiento con optimización bayesiana...")

    # Cargar parámetros de LightGBM desde YAML
    lgb_params = parametros["lgb_param_BO"]

    # Extraer rangos para optimización
    optimization_params = Dict(
        "learning_rate" => lgb_params["learning_rate"],
        "feature_fraction" => lgb_params["feature_fraction"],
        "num_leaves" => lgb_params["num_leaves"],
        "min_data_in_leaf" => lgb_params["min_data_in_leaf"]
    )

    # Configurar modelo de MLJ para LightGBM

   
    model = LGBMClassification(; lgb_params...)

    # Crear la tarea de predicción
    task_X = dataset_bo[:, Not(:clase_ternaria)]
    task_y = dataset_bo.clase_ternaria
    task = machine(model, task_X, task_y)

    # Definir los rangos para la optimización bayesiana
    ranges = NamedTuple(
        "learning_rate" => range(model, :learning_rate, lower=optimization_params["learning_rate"][1], upper=optimization_params["learning_rate"][2]),
        "feature_fraction" => range(model, :feature_fraction, lower=optimization_params["feature_fraction"][1], upper=optimization_params["feature_fraction"][2]),
        "num_leaves" => range(model, :num_leaves, lower=optimization_params["num_leaves"][1], upper=optimization_params["num_leaves"][2], scale=:log),
        "min_data_in_leaf" => range(model, :min_data_in_leaf, lower=optimization_params["min_data_in_leaf"][1], upper=optimization_params["min_data_in_leaf"][2], scale=:log)
    )

    # Ruta para el archivo de log
    log_file_path = joinpath(param_local["experimento"], "log_bo.txt")

    # Función para registrar los resultados en cada iteración
    function log_iteration(params, score)
        open(log_file_path, "a") do io
            println(io, "Parametros: ", params, " | Score: ", score)
        end
    end

    # Configurar estrategia de optimización bayesiana
    tuning = TunedModel(
        model=model,
        resampling=CV(nfolds=5),
        range=ranges,
        measure=MLJ.binary_log_loss,
        tuning=RandomSearch(max_attempts=100),
        logger=log_iteration
    )

    # Entrenar modelo con optimización
    machine_tuning = machine(tuning, task_X, task_y)
    fit!(machine_tuning)

    # Mostrar mejores hiperparámetros
    println("Mejores hiperparámetros:")
    println(report(machine_tuning))

    return machine_tuning
end






######################################################################################

# Definir la función de entrenamiento
function entrenar(modelo, X, y)
    # Cargar los hiperparámetros en el modelo
    modelo.boosting = param_local["lgb_param"]["boosting"]
    modelo.objective = param_local["lgb_param"]["objective"]
    modelo.metric = param_local["lgb_param"]["metric"]
    #modelo.first_metric_only = param_local["lgb_param"]["first_metric_only"]
    modelo.boost_from_average = param_local["lgb_param"]["boost_from_average"]
    modelo.feature_pre_filter = param_local["lgb_param"]["feature_pre_filter"]
    modelo.force_row_wise = param_local["lgb_param"]["force_row_wise"]
    modelo.max_depth = param_local["lgb_param"]["max_depth"]
    modelo.min_gain_to_split = param_local["lgb_param"]["min_gain_to_split"]
    modelo.min_sum_hessian_in_leaf = param_local["lgb_param"]["min_sum_hessian_in_leaf"]
    modelo.lambda_l1 = param_local["lgb_param"]["lambda_l1"]
    modelo.lambda_l2 = param_local["lgb_param"]["lambda_l2"]
    modelo.max_bin = param_local["lgb_param"]["max_bin"]
    modelo.num_iterations = param_local["lgb_param"]["num_iterations"]
    modelo.bagging_fraction = param_local["lgb_param"]["bagging_fraction"]
    modelo.pos_bagging_fraction = param_local["lgb_param"]["pos_bagging_fraction"]
    modelo.neg_bagging_fraction = param_local["lgb_param"]["neg_bagging_fraction"]
    modelo.is_unbalance = param_local["lgb_param"]["is_unbalance"]
    modelo.scale_pos_weight = param_local["lgb_param"]["scale_pos_weight"]
    modelo.drop_rate = param_local["lgb_param"]["drop_rate"]
    modelo.max_drop = param_local["lgb_param"]["max_drop"]
    modelo.skip_drop = param_local["lgb_param"]["skip_drop"]
    modelo.extra_trees = param_local["lgb_param"]["extra_trees"]
    modelo.learning_rate = param_local["lgb_param"]["learning_rate"]
    modelo.feature_fraction = param_local["lgb_param"]["feature_fraction"]
    modelo.num_leaves = param_local["lgb_param"]["num_leaves"]
    modelo.min_data_in_leaf = param_local["lgb_param"]["min_data_in_leaf"]
    modelo.num_class = 1

    # Convertir valores faltantes en X a 0
    X = replace(X, missing => 0.0)

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
        @info "Entrenando el modelo  - $(now())"
        fit!(modelo, X, y, verbosity=-1)
    catch e
        println("Error durante el entrenamiento: ", e)
    end
end

# Definir la función de predicción
function predecir(modelo, X)
    X = replace(X, missing => 0.0)
    return predict(modelo, X)
end

#######################################################################################
## Cargar Archivos a Kaggle
function convertir_df(submissions_output)
    # Divide la salida en líneas
    lines = split(submissions_output, "\n")

    # Ignora las primeras 3 líneas (encabezado)
    lines = lines[3:11]

    # Crea un vector de NamedTuples para almacenar los datos
    submissions = []

    # Itera sobre las líneas y extrae los datos
    for line in lines
        # Divide la línea en campos
        campos = split(line, r"\s+")

        # Extrae los datos
        fileName = campos[1]
        date_time = join(campos[2:3], " ")
        date_time = DateTime(date_time, "yyyy-mm-dd HH:MM:SS")

        # Busca el campo "complete" y el valor de publicScore
        status_index = findfirst(x -> x == "complete", campos)
        if status_index !== nothing
            status = campos[status_index]
            publicScore = parse(Float64, campos[status_index+1])
        else
            status = missing
            publicScore = missing
        end

        # Agrega el registro al vector
        push!(submissions, (fileName, date_time, status, publicScore))
    end

    # Crea un DataFrame a partir del vector
    df = DataFrame(submissions, [:fileName, :date_time, :status, :publicScore])
    return df
end

function cargar_y_obtener_ganancia(filepath::String, competition::String)
    # Enviar el archivo a Kaggle y obtener el submissionId
    submit_command = `kaggle competitions submit -c $competition -f $filepath -m "Carga automática desde Julia"`
    submit_output = read(submit_command, String)

    # Esperar unos segundos antes de consultar la puntuación
    sleep(15)

    # Obtener todos los envíos de la competencia
    score_command = `kaggle competitions submissions -c $competition`
    submissions_output = read(score_command, String)

    # Convertir a DataFrame
    submissions = convertir_df(submissions_output)

    # Verificar que el DataFrame no esté vacío
    if nrow(submissions) == 0
        println("No se encontraron envíos en la competencia.")
        return nothing
    end

    sort!(submissions, :date_time, rev=true)

    # Obtener el puntaje del envío más reciente
    latest_submission = submissions[1, :]
    publicScore = latest_submission.publicScore

    return publicScore
end



#######################################################################################
## FUuncion para generar cortes a partir de las predicciones del modelo.

function generar_csv_cortes(predicciones::DataFrame)
    # Verificar que el DataFrame `predicciones` tenga las columnas requeridas
    @info "Tamaño de predicciones", size(predicciones)
    @info "Tipo de archivo", typeof(predicciones)
    println(first(predicciones, 5))  # Imprime las primeras 5 filas del DataFrame

    if !all(in(["numero_de_cliente", "Predicted"], names(predicciones)))
        @info("El DataFrame debe contener las columnas `numero_de_cliente` y `Predicted`.")
    end

    @info "Ordenado de Predicciones"
    # Ordenar el DataFrame de mayor a menor según la columna `Predicted`
    sort!(predicciones, :Predicted, rev=true)

    @info "Guardado de Predicciones ordenadas."
    # Guardar las predicciones ordenadas en un archivo .txt
    nombre_archivo_predicciones = joinpath(param_local["experimento"], "predicciones_ordenadas.txt")
    #nombre_archivo_predicciones = "D:\\Backup DMEYF\\expe_julia\\predicciones_ordenadas.txt"
    open(nombre_archivo_predicciones, "w") do io
        for (i, row) in enumerate(eachrow(predicciones))
            println(io, "Cliente $(row.numero_de_cliente): $(row.Predicted)")
        end
    end
    println("Archivo generado: $nombre_archivo_predicciones")

    # Definir los cortes de 8500 a 13500, sumando de a 500
    cortes = 9000:500:11500

    @info "Generacion de cortes para kaggle"
    # Generar CSV para cada corte
    for corte in cortes
        # Crear copia de `predicciones` y asignar 1 para los primeros `corte` y 0 para el resto
        resultados_corte = copy(predicciones)
        resultados_corte.Predicted .= 0
        resultados_corte[1:corte, :Predicted] .= 1

        # Guardar el archivo CSV con el nombre correspondiente al corte
        nombre_archivo = joinpath(param_local["experimento"], "predicciones_corte_$corte.csv")

        #nombre_archivo = "D:\\Backup DMEYF\\expe_julia\\predicciones_corte_$corte.csv"

        CSV.write(nombre_archivo, resultados_corte; header=["numero_de_cliente", "Predicted"])
        println("Archivo generado: $nombre_archivo")

        @info "Carga de Archivo a Kaggle", corte
        ganancia = cargar_y_obtener_ganancia(nombre_archivo, param_local["competencia_kaggle"])
        println("Ganancia del submit: ", ganancia)

    end
end

