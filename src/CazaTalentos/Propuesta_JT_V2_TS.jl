using Random
using Distributions
using DataFrames
using Base.Threads  # Para utilizar hilos

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
    return [ftirar(GLOBAL_gimnasio[:jugadoras][pid], pcantidad) for pid in pids]
end

# Veredicto final
function gimnasio_veredicto(jugadora_id)
    return Dict(
        "tiros_total" => GLOBAL_gimnasio[:tiros_total],
        "acierto" => Int(jugadora_id == GLOBAL_gimnasio[:mejor_jugadora_id])
    )
end

# Implementación de Thompson Sampling 
# Implementación sin distribuciones Beta, solo con selección por rondas basada en resultados
function estrategia_mejorada_sin_beta(desviacion_corte=0.0)
    gimnasio_init()  # Inicializar el gimnasio

    # Inicializar la planilla con los encestes y estados iniciales de las jugadoras
    num_jugadoras = length(GLOBAL_gimnasio[:jugadoras])
    encestes = zeros(Int, num_jugadoras)  # Contador de encestes acumulados para cada jugadora
    activa = trues(num_jugadoras)  # Vector booleano que indica si la jugadora sigue activa
    mejor_jugadora_id = GLOBAL_gimnasio[:mejor_jugadora_id]  # Guardar el índice de "taurasita"
    taurasita_eliminada_primera_ronda = false  # Variable para rastrear si taurasita es eliminada en la primera ronda

    # Ronda Inicial con 3 Tiros
    for i in 1:num_jugadoras
        encestes[i] = ftirar(GLOBAL_gimnasio[:jugadoras][i], 50)
    end

    # Verificar si taurasita es eliminada después de la primera ronda usando una distribución normal
    media_encestes = mean(encestes)
    desviacion_encestes = std(encestes)

    umbral = media_encestes + desviacion_corte * desviacion_encestes  # Ajuste basado en la desviación estándar

    # Revisar si taurasita es eliminada
    if encestes[mejor_jugadora_id] < umbral
        taurasita_eliminada_primera_ronda = true
    end

    # Seleccionar las jugadoras activas para realizar tiros
    jugadoras_activas = findall(activa)

    # Actualizar si las jugadoras siguen activas o no
    for jugadora in jugadoras_activas
        if encestes[jugadora] < umbral
            activa[jugadora] = false  # Marcar como inactiva
        end
    end

    # Bucle de rondas adaptativas
    max_rondas = 100
    tiros_por_ronda_base = 50

    for ronda in 1:max_rondas
        # Verificar si la mejor jugadora sigue activa
        if !activa[mejor_jugadora_id]
            println("La mejor jugadora (taurasita) fue eliminada en la ronda $ronda")
            break
        end

        # Ajuste adaptativo de tiros por ronda
        tiros_por_ronda = tiros_por_ronda_base * (ronda // 100 + 1)

        # Seleccionar las jugadoras activas para realizar tiros
        jugadoras_activas = findall(activa)

        # Realizar los tiros para las jugadoras activas
        resultados_ronda = gimnasio_tirar(jugadoras_activas, tiros_por_ronda)

        # Actualizar encestes acumulados para las jugadoras activas
        for (idx, jugadora) in enumerate(jugadoras_activas)
            encestes[jugadora] += resultados_ronda[idx]
        end

        # Obtener el valor máximo entre las jugadoras activas
        mejor_valor = maximum([encestes[j] for j in jugadoras_activas])

        # Ajustar la actividad de las jugadoras (actualizando sus prioridades) usando la desviación estándar
        media_encestes = mean([encestes[j] for j in jugadoras_activas])
        desviacion_encestes = std([encestes[j] for j in jugadoras_activas])
        umbral = media_encestes + desviacion_corte * desviacion_encestes  # Ajuste basado en la desviación estándar

        # Actualizar si las jugadoras siguen activas o no
        for jugadora in jugadoras_activas
            if encestes[jugadora] < umbral
                activa[jugadora] = false  # Marcar como inactiva
            end
        end

        # Verificar si hemos encontrado con suficiente certeza la mejor jugadora
        if activa[mejor_jugadora_id] && encestes[mejor_jugadora_id] >= mejor_valor - 1
            # Consideramos que hemos encontrado a la mejor jugadora con alta probabilidad
            break
        end
    end

    # Obtener el veredicto
    return Dict("veredicto" => gimnasio_veredicto(findfirst(activa)), "taurasita_eliminada_primera_ronda" => taurasita_eliminada_primera_ronda)
end

# Ejecutar la estrategia con un número determinado de repeticiones (multi-hilo)
function ejecutar_estrategia_multihilo_sin_beta(n_repeticiones, desviacion_corte=0.0)
    aciertos = Atomic{Int}(0)  # Usamos variables atómicas para evitar conflictos de escritura entre hilos
    tiros_totales = Atomic{Int}(0)
    eliminaciones_primera_ronda = Atomic{Int}(0)  # Contador de cuántas veces se elimina taurasita en la primera ronda

    @threads for i in 1:n_repeticiones
        resultado = estrategia_mejorada_sin_beta(desviacion_corte)
        veredicto = resultado["veredicto"]
        if resultado["taurasita_eliminada_primera_ronda"]
            atomic_add!(eliminaciones_primera_ronda, 1)
        end
        atomic_add!(aciertos, round(Int, veredicto["acierto"]))  # Suma atómica para evitar errores de concurrencia, redondeando a Int
        atomic_add!(tiros_totales, round(Int, veredicto["tiros_total"]))
    end

    tasa_acierto = aciertos[] / n_repeticiones
    tiros_promedio = tiros_totales[] / n_repeticiones
    eliminaciones_totales = eliminaciones_primera_ronda[]

    println("La tasa de elección de la verdadera mejor es: ", tasa_acierto)
    println("La cantidad de tiros promedio en lograrlo es: ", tiros_promedio)
    println("La cantidad de veces que taurasita fue eliminada en la primera ronda es: ", eliminaciones_totales)
end

# Ejecutar la estrategia
Random.seed!(214363)  # Fijar semilla para reproducibilidad
n_repeticiones = 100000  # Número de simulaciones
@time ejecutar_estrategia_multihilo_sin_beta(n_repeticiones, -1)

