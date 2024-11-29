using Random
using Statistics
using DataFrames
using Primes

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
    GLOBAL_gimnasio[:taurasita] = taurasita
    # Crear otras jugadoras con probabilidades entre 0.204 y 0.400
    otras_jugadoras = collect(0.204:0.002:0.400)
    # Combinar todas las jugadoras y barajarlas
    jugadoras = shuffle(vcat(otras_jugadoras, GLOBAL_gimnasio[:taurasita]...))
    GLOBAL_gimnasio[:jugadoras] = jugadoras
    GLOBAL_gimnasio[:tiros_total] = 0
    # Identificar la posición de la mejor jugadora (Taurasita)
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

# Estrategia Mejorada: Successive Halving
function estrategia_mejorada_successive_halving(eta, tiros_por_ronda_inicial, max_rondas_final, tiro_ultima_ronda)
    gimnasio_init()  # Inicializar el gimnasio

    # Inicializar variables
    num_jugadoras = length(GLOBAL_gimnasio[:jugadoras])
    encestes = zeros(Int, num_jugadoras)  
    activa = trues(num_jugadoras)  
    
    ronda_num = 1
    tiros_por_ronda = tiros_por_ronda_inicial
    eta_factor = eta

    while true
        jugadoras_activas = findall(activa)
        num_activas = length(jugadoras_activas)
        
        # Si solo queda una jugadora, terminar
        if num_activas == 1
            break
        end
        
        # Realizar tiros para las jugadoras activas
        if ronda_num > 1
            tiros_por_ronda = tiros_por_ronda + ((ronda_num-1)*5)
        end

        resultados_ronda = gimnasio_tirar(jugadoras_activas, tiros_por_ronda)
        
        # Actualizar los encestes acumulados
        for (idx, jugadora) in enumerate(jugadoras_activas)
            encestes[jugadora] += resultados_ronda[idx]
        end
        
        # Calcular la tasa de acierto para cada jugadora activa
        total_tiros_actual = tiros_por_ronda * ronda_num
        tasas = [encestes[j] / total_tiros_actual for j in jugadoras_activas]
        
        # Ordenar las jugadoras activas por su tasa de acierto descendente
        ordenadas = sortperm(tasas, rev=true)
        jugadoras_ordenadas = jugadoras_activas[ordenadas]
        
        # Determinar el número de jugadoras a mantener
        k = ceil(Int, num_activas / eta_factor)
        k = max(1, k)  # Asegurar al menos una jugadora
        
        # Eliminar las jugadoras con menor desempeño
        jugadoras_a_eliminar = jugadoras_ordenadas[k+1:end]
        for j in jugadoras_a_eliminar
            activa[j] = false
        end
        
        #println("Ronda $ronda_num: Quedan $(count(activa)) jugadoras activas después de eliminar $(length(jugadoras_a_eliminar)) jugadoras.")
        
        # Incrementar la ronda y ajustar tiros por ronda si es necesario
        ronda_num += 1
        if ronda_num > max_rondas_final
            break
        end
    end
    
    # Ronda Final: Realizar tiros adicionales para confirmar
    jugadoras_activas = findall(activa)
    resultados_ronda_final = gimnasio_tirar(jugadoras_activas, tiro_ultima_ronda)
    
    # Actualizar los encestes acumulados
    for (idx, jugadora) in enumerate(jugadoras_activas)
        encestes[jugadora] += resultados_ronda_final[idx]
    end
    
    # Seleccionar la mejor jugadora basada en la cantidad de encestes
    mejor_jugadora_id = argmax(encestes)
    
    # Obtener el veredicto
    return gimnasio_veredicto(mejor_jugadora_id)
end

# Ejecutar la estrategia con Successive Halving
function ejecutar_estrategia_successive_halving(n_repeticiones, eta, tiros_por_ronda_inicial, max_rondas_final, tiro_ultima_ronda)
    aciertos = 0
    tiros_totales = 0
    
    for _ in 1:n_repeticiones
        veredicto = estrategia_mejorada_successive_halving(eta, tiros_por_ronda_inicial, max_rondas_final, tiro_ultima_ronda)
        aciertos += veredicto["acierto"]
        tiros_totales += veredicto["tiros_total"]
    end
    
    tasa_acierto = aciertos / n_repeticiones
    tiros_promedio = tiros_totales / n_repeticiones
    
    println("La tasa de elección de la verdadera mejor es: ", tasa_acierto)
    println("La cantidad de tiros promedio en lograrlo es: ", tiros_promedio)
    return tasa_acierto, tiros_promedio
end

# Parámetros del algoritmo Successive Halving
eta = 1.7  # Factor de reducción, usualmente 2
tiros_por_ronda_inicial = 30  # Número de tiros por ronda inicial
max_rondas_final = 5  # Número máximo de rondas antes de la ronda final
tiro_ultima_ronda = 300 # Tiros en la ronda final para confirmar

# Ejecutar la estrategia
function main_successive_halving()
    initial_seed = 214363
    Random.seed!(initial_seed)  # Fijar semilla para reproducibilidad
    
    n_repeticiones = 100000  # Número de simulaciones
    repeticionesExp = 1  # Número de repeticiones del experimento con diferentes semillas
    
    semillas = shuffle(primes(100000,999999))[1:repeticionesExp]
    
    global AUX1 = 0
    
    for experimento in 1:repeticionesExp
        Random.seed!(semillas[experimento])  # Variar la semilla con cada repetición usando un número primo inicial
        @time tasa_acierto, tiros_promedio = ejecutar_estrategia_successive_halving(n_repeticiones, eta, tiros_por_ronda_inicial, max_rondas_final, tiro_ultima_ronda)
        
        if tasa_acierto >= 0.99
            global AUX1 +=1
        end
    end
    println("De ", repeticionesExp, " repeticiones, resultaron superiores a 0.99 de ratio: ",  AUX1)
end

# Ejecutar el algoritmo
main_successive_halving()

