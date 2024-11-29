using Random
using Statistics
using DataFrames
using Base.Threads
using CSV


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
    #peloton = collect(0.204:0.002:0.400)  # 99 jugadoras
    #jugadoras = shuffle(append!(peloton, taurasita))  # Mezclar las jugadoras
    #GLOBAL_gimnasio[:jugadoras] = jugadoras
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
function estrategia_mejorada(desviacion_corte=0.0, tiros_por_ronda=25, sum_tiros=9, sum_desvio=0.05, corte_ronda=10, tiro_ultima_ronda=200)
    gimnasio_init()  # Inicializar el gimnasio

    # Inicializar la planilla con los encestes y estados iniciales de las jugadoras
    num_jugadoras = length(GLOBAL_gimnasio[:jugadoras])
    encestes = zeros(Int, num_jugadoras)  # Contador de encestes acumulados para cada jugadora
    activa = trues(num_jugadoras)  # Vector booleano que indica si la jugadora sigue activa

    ronda_num = 1
    
    # while ronda_num<=4 
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

    # Ronda Final: Realizar 200 tiros para las jugadoras restantes (cuando quedan 6 o menos)
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

# Ejecutar la estrategia con un número determinado de repeticiones (multi-hilo)
function ejecutar_estrategia_multihilo(n_repeticiones, desviacion_corte=0.0, tiros_por_ronda=25, sum_tiros=9, sum_desvio=0.05, corte_ronda=10, tiro_ultima_ronda=200)
    aciertos = Atomic{Int}(0)  # Usamos variables atómicas para evitar conflictos de escritura entre hilos
    tiros_totales = Atomic{Int}(0)

    
    for _ in 1:n_repeticiones
        veredicto = estrategia_mejorada(desviacion_corte, tiros_por_ronda, sum_tiros, sum_desvio, corte_ronda, tiro_ultima_ronda)
        atomic_add!(aciertos, veredicto["acierto"])  # Suma atómica para evitar errores de concurrencia
        atomic_add!(tiros_totales, veredicto["tiros_total"])
    end

    tasa_acierto = aciertos[] / n_repeticiones
    tiros_promedio = tiros_totales[] / n_repeticiones

    println("La tasa de elección de la verdadera mejor es: ", tasa_acierto)
    println("La cantidad de tiros promedio en lograrlo es: ", tiros_promedio)
    return tasa_acierto, tiros_promedio
end

# Ejecutar la estrategia
Random.seed!(214363)  # Fijar semilla para reproducibilidad

n_repeticiones = 100000  # Número de simulaciones

#@time tasa_acierto, tiros_promedio = ejecutar_estrategia_multihilo(n_repeticiones, 0.2, 50, 15, 0.07, 10, 200)


# Definir los valores para cada parámetro que queremos explorar
desviacion_corte_vals = -0.5:0.1:0.5
tiros_por_ronda_vals = 30:5:60
sum_tiros_vals = 5:2:15
sum_desvio_vals = 0.01:0.01:0.1
corte_ronda_vals = 5:5:20
tiro_ultima_ronda_vals = 100:50:300


# Definir nombre del archivo de resultados
result_file = "resultados_grid_search_100000.csv"

# Inicializar variables para almacenar los resultados
global mejor_tasa_acierto = 0.0
global mejor_tiros_promedio = typemax(Int)
global mejor_parametros = nothing

# Crear un DataFrame para almacenar los resultados si el archivo ya existe
if isfile(result_file)
    println("El archivo de resultados ya existe. Cargando y volviendo a evaluar...")
    # Leer el archivo de resultados existente
    df_resultados = CSV.read(result_file, DataFrame)

    # Re-ejecutar cada combinación de parámetros previamente almacenados
    for row in eachrow(df_resultados)
        desviacion_corte = row.desviacion_corte
        tiros_por_ronda = row.tiros_por_ronda
        sum_tiros = row.sum_tiros
        sum_desvio = row.sum_desvio
        corte_ronda = row.corte_ronda
        tiro_ultima_ronda = row.tiro_ultima_ronda

        # Ejecutar la estrategia con los parámetros actuales
        tasa_acierto, tiros_promedio = ejecutar_estrategia_multihilo(
            n_repeticiones,  # Menos iteraciones para ahorrar tiempo en la evaluación
            desviacion_corte,
            tiros_por_ronda,
            sum_tiros,
            sum_desvio,
            corte_ronda,
            tiro_ultima_ronda
        )

        # Evaluar si supera la tasa de acierto de 0.99
        mejor = tasa_acierto > 0.99

        # Solo grabar resultados parciales en archivo CSV si la tasa_acierto > 0.99
        if mejor
            open(result_file, "a") do io
                println(io, "$desviacion_corte,$tiros_por_ronda,$sum_tiros,$sum_desvio,$corte_ronda,$tiro_ultima_ronda,$tasa_acierto,$tiros_promedio,true")
            end
        end
    end
else
    # Si el archivo no existe, inicializarlo con las cabeceras
    open(result_file, "w") do io
        println(io, "desviacion_corte,tiros_por_ronda,sum_tiros,sum_desvio,corte_ronda,tiro_ultima_ronda,tasa_acierto,tiros_promedio,mejor")
    end
end

# Grid Search
for desviacion_corte in desviacion_corte_vals
    for tiros_por_ronda in tiros_por_ronda_vals
        for sum_tiros in sum_tiros_vals
            for sum_desvio in sum_desvio_vals
                for corte_ronda in corte_ronda_vals
                    for tiro_ultima_ronda in tiro_ultima_ronda_vals
                        # Ejecutar la estrategia con los parámetros actuales
                        local tasa_acierto, tiros_promedio = ejecutar_estrategia_multihilo(
                            n_repeticiones,  # Menos iteraciones para ahorrar tiempo en la búsqueda inicial
                            desviacion_corte,
                            tiros_por_ronda,
                            sum_tiros,
                            sum_desvio,
                            corte_ronda,
                            tiro_ultima_ronda
                        )
                        
                        # Declarar las variables como globales dentro del bucle
                        global mejor_tasa_acierto
                        global mejor_tiros_promedio
                        global mejor_parametros

                        # Evaluar si supera la tasa de acierto de 0.99
                        mejor = tasa_acierto > 0.99

                        # Si además es el mejor resultado hasta el momento, actualizar variables
                        if mejor && tiros_promedio < mejor_tiros_promedio
                            mejor_tasa_acierto = tasa_acierto
                            mejor_tiros_promedio = tiros_promedio
                            mejor_parametros = (desviacion_corte, tiros_por_ronda, sum_tiros, sum_desvio, corte_ronda, tiro_ultima_ronda)
                        end

                        # Solo grabar resultados parciales en archivo CSV si la tasa_acierto > 0.99
                        if mejor
                            open(result_file, "a") do io
                                println(io, "$desviacion_corte,$tiros_por_ronda,$sum_tiros,$sum_desvio,$corte_ronda,$tiro_ultima_ronda,$tasa_acierto,$tiros_promedio,true")
                            end
                        end
                    end
                end
            end
        end
    end
end

# Al final, imprimir los mejores resultados encontrados
if mejor_parametros != nothing
    println("Mejores parámetros encontrados durante el Grid Search:")
    println("Desviación Corte: ", mejor_parametros[1])
    println("Tiros por Ronda: ", mejor_parametros[2])
    println("Suma Tiros: ", mejor_parametros[3])
    println("Suma Desvío: ", mejor_parametros[4])
    println("Corte Ronda: ", mejor_parametros[5])
    println("Tiros Última Ronda: ", mejor_parametros[6])
    println("Tasa de Acierto: ", mejor_tasa_acierto)
    println("Promedio de Tiros: ", mejor_tiros_promedio)
else
    println("No se encontraron parámetros que cumplan con tasa_acierto > 0.99")
end
