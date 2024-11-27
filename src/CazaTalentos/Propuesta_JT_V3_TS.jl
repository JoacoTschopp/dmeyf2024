using Random
using Distributions
using DataFrames
using Base.Threads

# Función auxiliar para los tiros
function ftirar(prob, qty)
    return sum(rand() < prob for _ in 1:qty)
end

# Variables globales para el gimnasio
GLOBAL_gimnasio = Dict()

# Inicializar el gimnasio
function gimnasio_init()
    # Definir las jugadoras
    taurasita = [0.5]
    peloton = collect(0.204:0.002:0.400)  # 99 jugadoras
    jugadoras = shuffle(append!(peloton, taurasita))  # Mezclar las jugadoras
    GLOBAL_gimnasio[:jugadoras] = jugadoras
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
function estrategia_mejorada(desviacion_corte=0.0)
    gimnasio_init()  # Inicializar el gimnasio

    # Inicializar la planilla con los encestes y estados iniciales de las jugadoras
    num_jugadoras = length(GLOBAL_gimnasio[:jugadoras])
    encestes = zeros(Int, num_jugadoras)  # Contador de encestes acumulados para cada jugadora
    activa = trues(num_jugadoras)  # Vector booleano que indica si la jugadora sigue activa

    ronda_num = 1
    while count(activa) > 6
        # Realizar tiros según la ronda
        tiros_por_ronda = 50  # Cantidad de tiros por ronda para mantener el azar bajo control
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

        ronda_num += 1
    end

    # Ronda Final: Realizar 200 tiros para las jugadoras restantes (cuando quedan 6 o menos)
    jugadoras_activas = findall(activa)
    resultados_ronda_final = gimnasio_tirar(jugadoras_activas, 200)

    # Actualizar los encestes acumulados para las jugadoras activas
    for (idx, jugadora) in enumerate(jugadoras_activas)
        encestes[jugadora] += resultados_ronda_final[idx]
    end

    # Seleccionar la mejor jugadora basada en la cantidad de encestes
    mejor_jugadora_id = argmax(encestes)

    # Obtener el veredicto
    return gimnasio_veredicto(mejor_jugadora_id)
end

# Ejecutar la estrategia con un número determinado de repeticiones (multi-hilo)
function ejecutar_estrategia_multihilo(n_repeticiones, desviacion_corte=0.0)
    aciertos = Atomic{Int}(0)  # Usamos variables atómicas para evitar conflictos de escritura entre hilos
    tiros_totales = Atomic{Int}(0)

    @threads for _ in 1:n_repeticiones
        veredicto = estrategia_mejorada(desviacion_corte)
        atomic_add!(aciertos, veredicto["acierto"])  # Suma atómica para evitar errores de concurrencia
        atomic_add!(tiros_totales, veredicto["tiros_total"])
    end

    tasa_acierto = aciertos[] / n_repeticiones
    tiros_promedio = tiros_totales[] / n_repeticiones

    println("La tasa de elección de la verdadera mejor es: ", tasa_acierto)
    println("La cantidad de tiros promedio en lograrlo es: ", tiros_promedio)
end

# Ejecutar la estrategia
Random.seed!(214363)  # Fijar semilla para reproducibilidad
n_repeticiones = 100000  # Número de simulaciones
@time ejecutar_estrategia_multihilo(n_repeticiones, -1)


