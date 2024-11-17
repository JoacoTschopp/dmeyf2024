# Script: estrategia_optima.jl
using Random
using Statistics
using DataFrames

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
    peloton = collect(0.204:0.002:0.400)  # 99 jugadoras
    jugadoras = shuffle(append!(peloton, taurasita))  # Mezclar las jugadoras
    GLOBAL_gimnasio[:jugadoras] = jugadoras
    GLOBAL_gimnasio[:tiros_total] = 0
    GLOBAL_gimnasio[:mejor_jugadora_id] = findall(x -> x == 0.5, jugadoras)[1]
end

# Función para realizar los tiros
function gimnasio_tirar(pids, pcantidad)
    GLOBAL_gimnasio[:tiros_total] += length(pids) * pcantidad
    return [ftirar(GLOBAL_gimnasio[:jugadoras][id], pcantidad) for id in pids]
end

# Veredicto final
function gimnasio_veredicto(jugadora_id)
    return Dict(
        "tiros_total" => GLOBAL_gimnasio[:tiros_total],
        "acierto" => Int(jugadora_id == GLOBAL_gimnasio[:mejor_jugadora_id])
    )
end

# Función para realizar una ronda eliminatoria
function ronda_eliminatoria!(planilla, tiros_por_jugadora, num_jugadoras_a_pasar)
    # Obtener las jugadoras activas
    jugadoras_activas = planilla[planilla.activa.==1, :]
    if nrow(jugadoras_activas) == 0
        return
    end

    # Realizar los tiros
    ids_juegan = jugadoras_activas.id
    resultados = gimnasio_tirar(ids_juegan, tiros_por_jugadora)
    planilla[planilla.id.∈ids_juegan, :encestes] .+= resultados

    # Ordenar por encestes acumulados
    planilla_ordenada = sort(planilla[planilla.activa.==1, :], by=:encestes, rev=true)

    # Seleccionar las mejores jugadoras para pasar a la siguiente ronda
    if nrow(planilla_ordenada) > num_jugadoras_a_pasar
        ids_eliminadas = planilla_ordenada.id[num_jugadoras_a_pasar+1:end]
        planilla[planilla.id.∈ids_eliminadas, :activa] .= 0
    end
end

# Estrategia Optimizada
function estrategia_optimizada()
    gimnasio_init()  # Inicializar el gimnasio

    # Crear la planilla de la cazatalentos con todas las jugadoras activas
    planilla = DataFrame(
        id=collect(1:100),  # IDs de las jugadoras
        activa=ones(Int, 100),  # Todas las jugadoras están activas
        encestes=zeros(Int, 100)  # Encestes acumulados
    )

    # Ronda 1
    tiros_ronda1 = 20
    num_jugadoras_ronda1 = 50  # Pasan las mejores 50
    ronda_eliminatoria!(planilla, tiros_ronda1, num_jugadoras_ronda1)

    # Ronda 2
    tiros_ronda2 = 30
    num_jugadoras_ronda2 = 20  # Pasan las mejores 20
    ronda_eliminatoria!(planilla, tiros_ronda2, num_jugadoras_ronda2)

    # Ronda 3
    tiros_ronda3 = 50
    num_jugadoras_ronda3 = 5  # Pasan las mejores 5
    ronda_eliminatoria!(planilla, tiros_ronda3, num_jugadoras_ronda3)

    # Ronda Final
    tiros_ronda_final = 200
    ronda_eliminatoria!(planilla, tiros_ronda_final, 1)  # Seleccionamos a la mejor

    # Obtener la jugadora seleccionada
    jugadora_seleccionada = planilla[planilla.activa.==1, :id][1]

    # Obtener el veredicto
    return gimnasio_veredicto(jugadora_seleccionada)
end

# Ejecución de la estrategia y cálculo de métricas
function ejecutar_estrategia(n_repeticiones)
    aciertos = 0
    tiros_totales = 0

    for i in 1:n_repeticiones
        resultado = estrategia_optimizada()
        aciertos += resultado["acierto"]
        tiros_totales += resultado["tiros_total"]
    end

    tasa_acierto = aciertos / n_repeticiones
    tiros_promedio = tiros_totales / n_repeticiones

    println("La tasa de elección de la verdadera mejor es: ", tasa_acierto)
    println("La cantidad de tiros promedio en lograrlo es: ", tiros_promedio)
end

# Ejecutar la estrategia con un número determinado de repeticiones
Random.seed!(214363)  # Fijar semilla para reproducibilidad
n_repeticiones = 100000  # Número de simulaciones
@time ejecutar_estrategia(n_repeticiones)
