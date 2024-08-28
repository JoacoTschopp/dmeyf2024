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
Vn_subfeatures = UInt[0, 1, 2, 3]  # Número de subcaracterísticas (puede incluir 0 si no quieres subcaracterísticas)
Vmaxdepth = UInt[5, 7, 10, 15]  # Profundidad máxima del árbol
Vmin_samples_split = UInt[200, 400, 600, 800, 1000]  # Mínimo de muestras para dividir un nodo
Vmin_samples_leaf = UInt[10, 20, 30, 50]  # Mínimo de muestras por hoja
Vmin_purity_increase = Float64[0.0, 0.01, 0.02, 0.05]  # Mínimo incremento de pureza para realizar una división

training = 0.8
semilla = 27
qsemillas = 10 #0

### CALCULO GANANCIA

function  EstimarGanancia( psemilla, training, ptree )
    ganancia_test_normalizada = -1,0

    # particion
    Random.seed!(psemilla)
    vfold = 2 .-  Int.( rand(Float64, length(dataset_clase)) .< training )

    # train_campos =  replace!( Matrix( dataset[ vfold .== 1 ,  Not(:clase_ternaria)] ), missing => 0  )

    # clase = string.(dataset[ vfold .== 1 , :clase_ternaria ])
    # datos = Matrix( dataset[ vfold .== 1 ,  Not(:clase_ternaria)] )
    # genero el modelo en training
    modelo = DecisionTree.build_tree(
        dataset_clase[ vfold .== 1 ],
        dataset_matriz[ vfold .== 1 ,:],
        ptree.n_subfeatures,
        ptree.maxdepth,
        ptree.min_samples_leaf,
        ptree.min_samples_split,
        ptree.min_purity_increase
    )

    # aplico el modelo a testing vfold = 2
    pred = apply_tree_proba(modelo, 
        dataset_matriz[ vfold .== 2 ,:],
        ["BAJA+1","BAJA+2","CONTINUA"]
    )

    ganancia_test_normalizada = sum(
        ( dataset_clase[ vfold .== 2][ (pred[:, 2 ] .> 0.025) ] .== "BAJA+2"   )
        .* 280000
        .- 7000
       ) / ( 1.0 - training )
    
   return  ganancia_test_normalizada
end

####ENTRENAMIENTO Y CALCULO DE MEDIAS

# Función para evaluar la ganancia con diferentes combinaciones de hiperparámetros
function grid_search_hyperparameters(semillas, training, dataset_clase, dataset_matriz)
    mejor_ganancia = -Inf
    mejor_ptree = nothing

    @time @threads for n_subfeatures in Vn_subfeatures
        for maxdepth in Vmaxdepth
            for min_samples_split in Vmin_samples_split
                for min_samples_leaf in Vmin_samples_leaf
                    for min_purity_increase in Vmin_purity_increase
                        ptree = ttree(n_subfeatures, maxdepth, min_samples_split, min_samples_leaf, min_purity_increase)

                        # Vector para almacenar las ganancias obtenidas
                        ganancia = Array{Float64}(undef, length(semillas))

                        # Calcular las ganancias para cada semilla
                        @time @threads for i in 1:length(semillas)
                            ganancia[i] = EstimarGanancia(semillas[i], training, ptree)
                        end

                        # Calcular la media de las ganancias
                        ganancia_media = mean(ganancia)

                        # Si la ganancia media actual es mejor, actualizar los mejores hiperparámetros
                        if ganancia_media > mejor_ganancia
                            mejor_ganancia = ganancia_media
                            mejor_ptree = ptree
                        end

                        # Imprimir los resultados parciales para seguimiento
                        println("Hiperparámetros: $ptree")
                        @printf("Ganancia media: %.2f\n\n", ganancia_media)
                    end
                end
            end
        end
    end

    return mejor_ptree, mejor_ganancia
end

dataset = df

# restrinjo al periodo 202104
dataset = dataset[ dataset.foto_mes .== 202104, : ]

# Lamentablemente debo imputar nulos, 
#  porque la libreria DecisionTree no los soporta
dataset = Impute.substitute( dataset ) 

# formato para  DecisionTrees
dataset_clase = string.(dataset[ :, :clase_ternaria ])
dataset_matriz = Matrix( dataset[ :, Not(:clase_ternaria)] )

# elimino  dataset
dataset = Nothing

# genero la cantidad de qsemillas  nuevas semillas
Random.seed!(semilla)
semillas = rand( Primes.primes( 100000, 999999 ), qsemillas )

# Llamar a la función de búsqueda de hiperparámetros
mejor_ptree, mejor_ganancia = grid_search_hyperparameters(semillas, training, dataset_clase, dataset_matriz)

# Mostrar los mejores hiperparámetros encontrados
println("Mejores hiperparámetros encontrados: $mejor_ptree")
@printf("Mejor ganancia media: %.2f\n", mejor_ganancia)