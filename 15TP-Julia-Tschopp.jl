using CategoricalArrays
using CSV, DataFrames, Random, Statistics
using Primes
using DecisionTree, Impute
using Base.Threads
using Distributed
using Printf

df = CSV.read("G:/Mi unidad/01-Maestria Ciencia de Datos/DMEyF/TPs/dmeyf-2024/datasets/competencia_01_julia.csv", DataFrame)

### ARMADO DE ARBOL E HYPERPARAMETROS

struct ttree
    n_subfeatures::UInt
    maxdepth::UInt
    min_samples_split::UInt
    min_samples_leaf::UInt
    min_purity_increase::Float64
end

#ptree = ttree(0, 7, 800, 20, 0)

# Definir vectores para cada hiperparámetro
Vn_subfeatures = UInt[0, 1]  # Número de subcaracterísticas (puede incluir 0 si no quieres subcaracterísticas)
Vmaxdepth = UInt[5, 6, 7, 8]  # Profundidad máxima del árbol
Vmin_samples_split = UInt[400, 600, 800, 1000]  # Mínimo de muestras para dividir un nodo
Vmin_samples_leaf = UInt[10, 15, 20, 25, 30]  # Mínimo de muestras por hoja
Vmin_purity_increase = Float64[-1, -0.5, 0.0, 0.5, 1]  # Mínimo incremento de pureza para realizar una división

training = 0.7
semilla = 27
qsemillas = 5 #0

### CALCULO GANANCIA
function EstimarGanancia(psemilla, training, ptree)
    ganancia_test_normalizada = -1, 0

    Random.seed!(psemilla)
    vfold = 2 .- Int.(rand(Float64, length(dataset_clase)) .< training)

    modelo = DecisionTree.build_tree(
        dataset_clase[vfold.==1],
        dataset_matriz[vfold.==1, :],
        ptree.n_subfeatures,
        ptree.maxdepth,
        ptree.min_samples_leaf,
        ptree.min_samples_split,
        ptree.min_purity_increase
    )

    # aplico el modelo a testing vfold = 2
    pred = apply_tree_proba(modelo,
        dataset_matriz[vfold.==2, :],
        ["BAJA+1", "BAJA+2", "CONTINUA"]
    )

    ganancia_test_normalizada = sum(
        (dataset_clase[vfold.==2][(pred[:, 2].>0.025)] .== "BAJA+2")
        .*
        280000
        .-
        7000
    ) / (1.0 - training)

    return ganancia_test_normalizada
end

####ENTRENAMIENTO Y CALCULO DE MEDIAS

# Función para evaluar la ganancia con diferentes combinaciones de hiperparámetros
function grid_search_hyperparameters(semillas, training, dataset_clase, dataset_matriz, ruta_archivo)
    resultados = []

    @time @threads for n_subfeatures in Vn_subfeatures
        for maxdepth in Vmaxdepth
            for min_samples_split in Vmin_samples_split
                for min_samples_leaf in Vmin_samples_leaf
                    for min_purity_increase in Vmin_purity_increase
                        ptree = ttree(n_subfeatures, maxdepth, min_samples_split, min_samples_leaf, min_purity_increase)

                        ganancia = Array{Float64}(undef, length(semillas))

                        # Calcular las ganancias para cada semilla
                        @time @threads for i in 1:length(semillas)
                            ganancia[i] = EstimarGanancia(semillas[i], training, ptree)
                            if ganancia[i] == 0.0
                                break
                            end
                        end

                        # Guardar los hiperparámetros y ganancias en el archivo CSV
                        for g in ganancia
                            CSV.write(ruta_archivo, DataFrame(Dict(
                                :n_subfeatures => n_subfeatures,
                                :maxdepth => maxdepth,
                                :min_samples_split => min_samples_split,
                                :min_samples_leaf => min_samples_leaf,
                                :min_purity_increase => min_purity_increase,
                                :ganancia => g
                            )), append=true)
                        end

                        # Resultados parciales para seguimiento
                        println("Hiperparámetros: $ptree")
                        println("Ganancias: ", ganancia)
                    end
                end
            end
        end
    end

    # Convertir la lista de resultados en un DataFrame (opcional)
    df_resultados = DataFrame(resultados)
    return df_resultados
end

dataset = df

# restrinjo al periodo 202104
dataset = dataset[dataset.foto_mes.==202104, :]

# Lamentablemente debo imputar nulos, 
#  porque la libreria DecisionTree no los soporta
dataset = Impute.substitute(dataset)

# formato para  DecisionTrees
dataset_clase = string.(dataset[:, :clase_ternaria])
dataset_matriz = Matrix(dataset[:, Not(:clase_ternaria)])

# elimino  dataset
dataset = Nothing

# genero la cantidad de qsemillas  nuevas semillas
Random.seed!(semilla)
semillas = rand(Primes.primes(100000, 999999), qsemillas)

# Guardar el DataFrame como un archivo CSV
ruta_archivo = "G:/Mi unidad/01-Maestria Ciencia de Datos/DMEyF/TPs/dmeyf-2024/dmeyf2024/HiperparametrosWilcox/resultados-002.csv"
# Llamar a la función de búsqueda de hiperparámetros
df_resultados = grid_search_hyperparameters(semillas, training, dataset_clase, dataset_matriz, ruta_archivo)

println("Archivo guardado en: $ruta_archivo")