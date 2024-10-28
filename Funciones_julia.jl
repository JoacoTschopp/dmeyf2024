using LightGBM
using CSV
using DataFrames
using BayesianOptimization

#######################################################################################
#  BO
#######################################################################################
# Funciond e ganancia de la BO

function ganancia(df_predicciones::DataFrame, y_val)
    # Asegúrate de que `df_predicciones` tenga las columnas necesarias
    if !haskey(df_predicciones, :Predicted)
        error("La columna Predicted no se encontró en df_predicciones")
    end

    # Comparar las predicciones con los valores reales
    score = mean((y_val .- df_predicciones.Predicted).^2)  # Por ejemplo, error cuadrático medio
    return -score  # Retornamos el score negativo como ganancia positiva
end


# Función de optimización
"""
using BayesianOptimization, LightGBM, CSV, DataFrames

function optimizar_lightgbm(X_train, y_train, X_val, y_val, X_test, y_test, HP_fijos, HP_optimizar)
    # Archivos CSV para los logs
    log_file = "logs_lightgbm.csv"
    best_file = "mejores_resultados_lightgbm.csv"
    
    # Encabezado del archivo de logs
    CSV.write(log_file, DataFrame(Iteración=Int[], Ganancia=Float64[], Hiperparámetros=String[]), append=false)
    
    # Definir la función objetivo para optimización
    function objective(hp)
        # Combinar los hiperparámetros fijos y los optimizados
        params = merge(HP_fijos, hp)

        # Entrenar el modelo con LightGBM y los hiperparámetros actuales
        modelo = LGBMClassification(params...)
        fit!(modelo, X_train, y_train, verbosity = -1)
        
        # Obtener las predicciones del modelo en el conjunto de validación
        preds_val = predict(modelo, X_val)
        
        # Calcular la ganancia
        df_pred = DataFrame(numero_de_cliente = X_val[:, :numero_de_cliente], Predicted = preds_val)
        score_val = ganancia(df_pred, y_val)  # Llamada a la función ganancia

        # Guardar cada iteración en el archivo CSV
        new_row = DataFrame(Iteración=[hp[:iter]], Ganancia=[score_val], Hiperparámetros=[string(params)])
        CSV.write(log_file, new_row, append=true)
        
        return score_val  # La ganancia ya está en formato positivo
    end

    # Configuración para la optimización bayesiana
    opt_params = Dict(:num_samples => 50, :maximize => true)
    results = bayesopt(objective, HP_optimizar, Opts(; opt_params...))

    # Obtener el mejor resultado del log y guardarlo
    logs = CSV.read(log_file, DataFrame)
    mejor_resultado = logs[findmax(logs.Ganancia)[2], :]
    CSV.write(best_file, mejor_resultado)

    return mejor_resultado
end

"""


#######################################################################################
## FUuncion para generar cortes a partir de las predicciones del modelo.

function generar_csv_cortes(predicciones::DataFrame)
    # Verificar que el DataFrame `predicciones` tenga las columnas requeridas
    if !all(in(["numero_de_cliente", "Predicted"], names(predicciones)))
        error("El DataFrame debe contener las columnas `numero_de_cliente` y `Predicted`.")
    end

    @info "Ordenado de Predicciones"
    # Ordenar el DataFrame de mayor a menor según la columna `Predicted`
    sort!(predicciones, :Predicted, rev=true)

    @info "Guardado de Predicciones ordenadas."
    # Guardar las predicciones ordenadas en un archivo .txt
    nombre_archivo_predicciones = "~/buckets/b1/exportaJulia/predicciones_ordenadas.txt"
    open(nombre_archivo_predicciones, "w") do io
        for (i, row) in enumerate(eachrow(predicciones))
            println(io, "Cliente $(row.numero_de_cliente): $(row.Predicted)")
        end
    end
    println("Archivo generado: $nombre_archivo_predicciones")

    # Definir los cortes de 8500 a 13500, sumando de a 500
    cortes = 8500:500:13500

    @info "Generacion de cortes para kaggle"
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

