using Random
using Statistics
using DataFrames
using CSV
using Base.Threads

# Función auxiliar para los tiros
function ftirar(prob, qty)
    return sum(rand() < prob for _ in 1:qty)
end

# Inicializar el gimnasio
function gimnasio_init()
    # Definir las jugadoras
    taurasita = [0.5]
    jugadoras = shuffle(append!([0.204:0.002:0.400;], taurasita))
    tiros_total = 0
    mejor_jugadora_id = findall(x -> x == 0.5, jugadoras)[1]
    return Dict(
        :jugadoras => jugadoras,
        :tiros_total => tiros_total,
        :mejor_jugadora_id => mejor_jugadora_id
    )
end

# Función para realizar los tiros
function gimnasio_tirar(gimnasio, pids, pcantidad)
    gimnasio[:tiros_total] += length(pids) * pcantidad
    return [ftirar(gimnasio[:jugadoras][pid], pcantidad) for pid in pids]
end

# Veredicto final
function gimnasio_veredicto(gimnasio, jugadora_id)
    return Dict(
        "tiros_total" => gimnasio[:tiros_total],
        "acierto" => Int(jugadora_id == gimnasio[:mejor_jugadora_id])
    )
end

# Estrategia Mejorada
function estrategia_mejorada(desviacion_corte=0.0, tiros_por_ronda=25, sum_tiros=9, sum_desvio=0.05, corte_ronda=10, tiro_ultima_ronda=200)
    gimnasio = gimnasio_init()  # Inicializar el gimnasio

    # Inicializar la planilla con los encestes y estados iniciales de las jugadoras
    num_jugadoras = length(gimnasio[:jugadoras])
    encestes = zeros(Int, num_jugadoras)  # Contador de encestes acumulados para cada jugadora
    activa = trues(num_jugadoras)  # Vector booleano que indica si la jugadora sigue activa

    ronda_num = 1

    while count(activa) > corte_ronda
        # Realizar tiros según la ronda
        jugadoras_activas = findall(activa)
        resultados_ronda = gimnasio_tirar(gimnasio, jugadoras_activas, tiros_por_ronda)

        # Actualizar los encestes acumulados para las jugadoras activas
        for (idx, jugadora) in enumerate(jugadoras_activas)
            encestes[jugadora] += resultados_ronda[idx]
        end

        # Verificar y ajustar el umbral para eliminar jugadoras
        media_encestes = mean([encestes[j] for j in jugadoras_activas])
        desviacion_encestes = std([encestes[j] for j in jugadoras_activas])
        umbral = media_encestes + desviacion_corte * desviacion_encestes

        # Actualizar la actividad de las jugadoras (marcar como inactivas si están por debajo del umbral)
        for jugadora in jugadoras_activas
            if encestes[jugadora] < umbral
                activa[jugadora] = false
            end
        end

        # Incrementar la cantidad de tiros por ronda y ajustar el desvío
        tiros_por_ronda += sum_tiros
        desviacion_corte += sum_desvio

        ronda_num += 1
    end

    # Ronda Final
    jugadoras_activas = findall(activa)
    resultados_ronda_final = gimnasio_tirar(gimnasio, jugadoras_activas, tiro_ultima_ronda)

    # Actualizar los encestes acumulados para las jugadoras activas
    for (idx, jugadora) in enumerate(jugadoras_activas)
        encestes[jugadora] += resultados_ronda_final[idx]
    end

    # Seleccionar la mejor jugadora basada en la cantidad de encestes
    mejor_jugadora_id = argmax(encestes)

    # Obtener el veredicto
    return gimnasio_veredicto(gimnasio, mejor_jugadora_id)
end

# Ejecutar la estrategia con un número determinado de repeticiones
function ejecutar_estrategia(n_repeticiones, desviacion_corte=0.0, tiros_por_ronda=25, sum_tiros=9, sum_desvio=0.05, corte_ronda=10, tiro_ultima_ronda=200)
    aciertos = 0
    tiros_totales = 0

    for _ in 1:n_repeticiones
        veredicto = estrategia_mejorada(desviacion_corte, tiros_por_ronda, sum_tiros, sum_desvio, corte_ronda, tiro_ultima_ronda)
        aciertos += veredicto["acierto"]
        tiros_totales += veredicto["tiros_total"]
    end

    tasa_acierto = aciertos / n_repeticiones
    tiros_promedio = tiros_totales / n_repeticiones

    return tasa_acierto, tiros_promedio
end

# Definir los valores para cada parámetro que queremos explorar
desviacion_corte_vals = collect(-0.5:0.1:0.5)
tiros_por_ronda_vals = collect(30:5:60)
sum_tiros_vals = collect(5:2:15)
sum_desvio_vals = collect(0.01:0.01:0.1)
corte_ronda_vals = collect(5:5:20)
tiro_ultima_ronda_vals = collect(100:50:300)

# Definir nombre del archivo de resultados
result_file = "resultados_grid_search_paralelo.csv"

# Inicializar el archivo de resultados con las cabeceras si no existe
if !isfile(result_file)
    open(result_file, "w") do io
        println(io, "desviacion_corte,tiros_por_ronda,sum_tiros,sum_desvio,corte_ronda,tiro_ultima_ronda,tasa_acierto,tiros_promedio")
    end
end

# Número de simulaciones
n_repeticiones = 100000  # Número de simulaciones

# Crear el array de combinaciones de hiperparámetros
hyperparameter_combinations = [(dc, tpr, st, sd, cr, tur) for dc in desviacion_corte_vals,
                                                               tpr in tiros_por_ronda_vals,
                                                               st in sum_tiros_vals,
                                                               sd in sum_desvio_vals,
                                                               cr in corte_ronda_vals,
                                                               tur in tiro_ultima_ronda_vals]

# Función para simular y devolver resultados
function simulate_hyperparameters(params, n_repeticiones)
    desviacion_corte, tiros_por_ronda, sum_tiros, sum_desvio, corte_ronda, tiro_ultima_ronda = params
    tasa_acierto, tiros_promedio = ejecutar_estrategia(
        n_repeticiones,
        desviacion_corte,
        tiros_por_ronda,
        sum_tiros,
        sum_desvio,
        corte_ronda,
        tiro_ultima_ronda
    )
    return (desviacion_corte, tiros_por_ronda, sum_tiros, sum_desvio, corte_ronda, tiro_ultima_ronda, tasa_acierto, tiros_promedio)
end

# Ejecutar las simulaciones en paralelo usando hilos
@threads for i in 1:length(hyperparameter_combinations)
    params = hyperparameter_combinations[i]
    res = simulate_hyperparameters(params, n_repeticiones)
    # Guardar resultados si tasa_acierto > 0.99
    if res[7] > 0.99
        open(result_file, "a") do io
            println(io, join(res, ","))
        end
    end
end

# Encontrar los mejores parámetros
# Leer resultados del archivo CSV
if isfile(result_file)
    df = CSV.read(result_file, DataFrame)
    if nrow(df) > 0
        sorted_df = sort(df, :tasa_acierto, rev=true)
        best_result = first(sorted_df)
        println("Mejores parámetros encontrados durante el Grid Search:")
        println("Desviación Corte: ", best_result.desviacion_corte)
        println("Tiros por Ronda: ", best_result.tiros_por_ronda)
        println("Suma Tiros: ", best_result.sum_tiros)
        println("Suma Desvío: ", best_result.sum_desvio)
        println("Corte Ronda: ", best_result.corte_ronda)
        println("Tiros Última Ronda: ", best_result.tiro_ultima_ronda)
        println("Tasa de Acierto: ", best_result.tasa_acierto)
        println("Promedio de Tiros: ", best_result.tiros_promedio)
    else
        println("No se encontraron parámetros que cumplan con tasa_acierto > 0.99")
    end
else
    println("No se encontraron resultados previos.")
end
