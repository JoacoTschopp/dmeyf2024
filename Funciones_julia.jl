using LightGBM
using CSV
using DataFrames
#using BayesOpt

#######################################################################################
#  BO
#######################################################################################
# Función de optimización
"""
function optimizar_lightgbm(X_train, y_train, X_val, y_val, X_test, y_test, HP_fijos, HP_optimizar)
    # Crear archivo CSV para almacenar los resultados
    log_file = "logs_lightgbm.csv"
    best_file = "mejores_resultados_lightgbm.csv"

    # Escribir el encabezado del archivo CSV de log
    CSV.write(log_file, DataFrame(), append=false)
    
    # Definir la función objetivo
    function objective(hp)
        # Crear una configuración de hiperparámetros combinada con los valores fijos
        params = merge(HP_fijos, hp)
        
        # Entrenar el modelo con los parámetros actuales
        modelo = LGBMClassification()
        fit!(modelo, X_train, y_train, verbosity = -1)#, valid_set=(X_val, y_val)

        # Evaluar el modelo en el conjunto de validación
        preds_val = predict(modelo, X_val)

        # Usar una métrica de evaluación, como el error cuadrático medio
        score_val = mean((y_val .- preds_val).^2)
        
        # Guardar el resultado de la iteración en el archivo CSV
        new_row = DataFrame(; Iteración=[hp[:iter]], Ganancia=[-score_val], Hiperparámetros=[params])
        CSV.write(log_file, new_row, append=true)
        
        return -score_val  # Retorna ganancia negativa porque BO maximiza por defecto
    end

    # Ejecutar la optimización Bayesiana
    results = optimize(objective, HP_optimizar, max_evals=50)

    # Leer el log y guardar el mejor resultado en el archivo de resultados finales
    logs = CSV.read(log_file, DataFrame)
    mejor_resultado = logs[findmax(logs.Ganancia)[2], :]
    CSV.write(best_file, mejor_resultado)
    
    return mejor_resultado
end

# Ejemplo de uso de la función
X_train, y_train = "..." # Tus datos de entrenamiento
X_val, y_val = "..."     # Tus datos de validación
X_test, y_test =  "..."   # Tus datos de prueba

# Hiperparámetros fijos
HP_fijos = param_local["lgb_param"]

# Hiperparámetros que queremos optimizar
HP_optimizar = param_local["lgb_param_BO"]

# Llamar a la función de optimización
#resultado_final = optimizar_lightgbm(X_train, y_train, X_val, y_val, X_test, y_test, HP_fijos, HP_optimizar)

"""






#######################################################################################
## FUuncion para generar cortes a partir de las predicciones del modelo.

function generar_csv_cortes(predicciones::DataFrame)
    # Verificar que el DataFrame `predicciones` tenga las columnas requeridas
    if !all(["numero_de_cliente", "Predicted"] .∈ names(predicciones))
        error("El DataFrame debe contener las columnas `numero_de_cliente` y `Predicted`.")
    end

    # Ordenar el DataFrame de mayor a menor según la columna `Predicted`
    sort!(predicciones, :Predicted, rev=true)

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

