using Random
using Statistics
using DataFrames
using Primes 

# Función auxiliar para los tiros
function ftirar(prob, qty)
    return sum(rand() < prob for i in 1:qty)
end

# Variables globales para el gimnasio
GLOBAL_gimnasio = Dict()

# Inicializar el gimnasio
function gimnasio_init()
    # Definir las jugadoras
    taurasita = [0.5]
    GLOBAL_gimnasio[:taurasita] = taurasita
    GLOBAL_gimnasio[:jugadoras] = shuffle(append!([0.204:0.002:0.400;],
        GLOBAL_gimnasio[:taurasita]))
    jugadoras =  GLOBAL_gimnasio[:jugadoras]
    GLOBAL_gimnasio[:tiros_total] = 0
    GLOBAL_gimnasio[:mejor_jugadora_id] = findall(x -> x == 0.5, jugadoras)[1]
end

# Función para realizar los tiros
function gimnasio_tirar(pids, pcantidad)
    GLOBAL_gimnasio[:tiros_total] += length(pids) * pcantidad
    return [ftirar(GLOBAL_gimnasio[:jugadoras][pid], pcantidad) for pid in pids]
end

# Veredicto final
function gimnasio_veredicto(jugadora_id)
    return Dict(
        "tiros_total" => GLOBAL_gimnasio[:tiros_total],
        "acierto" => Int(jugadora_id == GLOBAL_gimnasio[:mejor_jugadora_id])
    )
end

# Estrategia Mejorada
function estrategia_mejorada(desviacion_corte, tiros_por_ronda, sum_tiros, sum_desvio, corte_ronda, tiro_ultima_ronda)
    gimnasio_init()  # Inicializar el gimnasio

    # Inicializar la planilla con los encestes y estados iniciales de las jugadoras
    num_jugadoras = length(GLOBAL_gimnasio[:jugadoras])
    encestes = zeros(Int, num_jugadoras)  
    activa = trues(num_jugadoras)  

    ronda_num = 1
    
    # Corta caundo quedan menos de lo indicado de participantes
    while count(activa) > corte_ronda 
        # Realizar tiros según la ronda
        jugadoras_activas = findall(activa)
        resultados_ronda = gimnasio_tirar(jugadoras_activas, tiros_por_ronda)

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

        # Incrementar la cantidad de tiros por ronda y reducir el corte del desvío
        tiros_por_ronda += sum_tiros
        desviacion_corte += sum_desvio
        ronda_num += 1
    end

    # Ronda Final: Realizar muchos tiros para las jugadoras restantes
    jugadoras_activas = findall(activa)
    resultados_ronda_final = gimnasio_tirar(jugadoras_activas, tiro_ultima_ronda)

    # Actualizar los encestes acumulados para las jugadoras activas
    for (idx, jugadora) in enumerate(jugadoras_activas)
        encestes[jugadora] += resultados_ronda_final[idx]
    end

    # Seleccionar la mejor jugadora basada en la cantidad de encestes
    mejor_jugadora_id = argmax(encestes)

    # Obtener el veredicto
    return gimnasio_veredicto(mejor_jugadora_id)
end

# Ejecutar la estrategia con un número aleatorio de repeticiones
function ejecutar_estrategia(n_repeticiones, desviacion_corte, tiros_por_ronda, sum_tiros, sum_desvio, corte_ronda, tiro_ultima_ronda)
    aciertos = 0
    tiros_totales = 0
    
    for _ in 1:n_repeticiones
        veredicto = estrategia_mejorada(desviacion_corte, tiros_por_ronda, sum_tiros, sum_desvio, corte_ronda, tiro_ultima_ronda)
        aciertos += veredicto["acierto"]
        tiros_totales += veredicto["tiros_total"]
    end

    tasa_acierto = aciertos[] / n_repeticiones
    tiros_promedio = tiros_totales[] / n_repeticiones

    println("La tasa de elección de la verdadera mejor es: ", tasa_acierto)
    println("La cantidad de tiros promedio en lograrlo es: ", tiros_promedio)
    return tasa_acierto, tiros_promedio
end

# Ejecutar la estrategia
initial_seed = 214363
Random.seed!(initial_seed)  # Fijar semilla para reproducibilidad 214363

n_repeticiones = 100000  # Número de simulaciones

global AUX1 = 0
repeticionesExp = 10 # Numero de repeticiones del experimento con la misma semilla

semillas = shuffle(primes(100000,999999))[1:repeticionesExp]

for experimento in 1:repeticionesExp
    Random.seed!(semillas[experimento])  # Variar la semilla con cada repetición usando un número primo inicial
    #Este tiene un ratio de validez del 50% de las veces con 10288 tiros de promedio
    @time tasa_acierto, tiros_promedio = ejecutar_estrategia(n_repeticiones, -0.4, 30, 5, 0.03, 5, 300) # Parametros Misteriosos, logrados claro a fuerza de CPU.
    
    #Este el 100% de las veces da positivo apra 0,99 de ratio de elegir a Taurasita con 10345 tiros de promedio
    #@time tasa_acierto, tiros_promedio = ejecutar_estrategia(n_repeticiones, -0.3, 35, 5, 0.02, 5, 260) # Parametros Misteriosos, logrados claro a fuerza de CPU.

    if tasa_acierto >= 0.99
        global AUX1 +=1
    end
end
println("De ", repeticionesExp, " repeticiones, resultaron superior a 0,99 de ratio: ",  AUX1)