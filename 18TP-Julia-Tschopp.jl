using CategoricalArrays
using CSV, DataFrames, Random, Statistics
using Primes
using DecisionTree, Impute
using Base.Threads
using Printf
using ZipFile
using StatsBase


df = CSV.read("G:/Mi unidad/01-Maestria Ciencia de Datos/DMEyF/TPs/dmeyf-2024/datasets/competencia_01_julia.csv", DataFrame)

###CALCULO GANANCIA
function EstimarGanancia(psemilla, dataset_train, dataset_202104, ptree, training = 0.8)
    # Semilla para reproducibilidad
    Random.seed!(psemilla)

    # Estratificación y selección aleatoria del 20% de los datos de 202104 para testing
    strata_indices = groupby(dataset_202104, :clase_ternaria)
    test_indices = vcat([rand(Random.default_rng(), 1:nrow(s), Int(round((1-training) * nrow(s)))) for s in strata_indices]...)
    train_indices = setdiff(1:nrow(dataset_202104), test_indices)

    # Crear subconjuntos de training y testing
    dataset_202104_train = dataset_202104[train_indices, :]
    dataset_202104_test = dataset_202104[test_indices, :]

    # Concatenar el 80% restante de 202104 con dataset_train
    full_train = vcat(dataset_train, dataset_202104_train)

    # Preparar los datos para el modelo
    full_train_clase = string.(full_train[:, :clase_ternaria])
    full_train_matriz = Matrix(full_train[:, Not(:clase_ternaria)])
    test_clase = string.(dataset_202104_test[:, :clase_ternaria])
    test_matriz = Matrix(dataset_202104_test[:, Not(:clase_ternaria)])

    # Construir el modelo
    modelo = DecisionTree.build_tree(
        full_train_clase,
        full_train_matriz,
        ptree.n_subfeatures,
        ptree.maxdepth,
        ptree.min_samples_leaf,
        ptree.min_samples_split,
        ptree.min_purity_increase
    )

    # Aplicar el modelo al conjunto de test
    pred = apply_tree_proba(modelo,
        test_matriz,
        ["BAJA+1", "BAJA+2", "CONTINUA"]
    )

    # Calcular la ganancia sobre el conjunto de test
    ganancia_test_normalizada = sum(
        (test_clase[(pred[:, 2] .> 0.025)] .== "BAJA+2") .* 280000 .- 7000
    )/ (1.0 - training)

    return ganancia_test_normalizada
end

# Definición de la estructura ttree para los hiperparámetros
struct ttree
    n_subfeatures::UInt
    maxdepth::UInt
    min_samples_split::UInt
    min_samples_leaf::UInt
    min_purity_increase::Float64
end

# Ejemplo de hiperparámetros
ptree = ttree(0, 5, 800, 10, -1.0)
##############################################
#
#La idea es sacar un 20% de los registros de 202104 para test.
#Se entrena con todo el resto pero esto lo hacemos en la funcion asi luego podemos llamarla reiteradamentes y sacar promedios
#
##############################################
# Filtrar dataset para el entrenamiento hasta el mes 202104
dataset_train = df[df.foto_mes .<= 202104, :]

# Imputar valores nulos y reemplazar por 0
dataset_train = coalesce.(Impute.substitute(dataset_train), 0)

# Extraer el subconjunto correspondiente a 202104
dataset_202104 = dataset_train[dataset_train.foto_mes .== 202104, :]

##############################################


# Conversión de clase_ternaria a string
#dataset_clase = string.(dataset_train[:, :clase_ternaria])
#dataset_matriz = Matrix(dataset_train[:, Not(:clase_ternaria)])

# Definir el DataFrame para 202104
#dataset_clase_202104 = string.(dataset_202104[:, :clase_ternaria])
#dataset_matriz_202104 = Matrix(dataset_202104[:, Not(:clase_ternaria)])

# Llamada a la función con los DataFrames completos



# Cálculo de la ganancia con partición aleatoria y estratificada del 20%
psemilla = 214363  # Puedes cambiar este valor para diferentes pruebas
qsemillas = 100
Random.seed!(psemilla)
semillas = rand(Primes.primes(100000, 999999), qsemillas)

ganancia = Array{Float64}(undef, length(semillas))

@time @threads for i = 1:length(semillas)
    ganancia[i], = EstimarGanancia(psemilla, dataset_train, dataset_202104, ptree)
end

# Imprimir la media con formato decimal
@printf("Media: %.2f\n", Statistics.mean(ganancia))
