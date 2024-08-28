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
 
 ptree = ttree(0, 7, 800, 20, 0)
 training = 0.8
 semilla = 27
 qsemillas = 100 #0

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


# vector donde almaceno los resultados
ganancia = Array{Float64}( undef, length( semillas ))

# calculo las  ganancias
@time @threads for i=1:length(semillas) 
    ganancia[i] = EstimarGanancia( semillas[i], training, ptree )
 end
  
# Imprimir cada valor en el array con formato decimal
for g in ganancia
    @printf("%.2f\n", g)
end

# Imprimir la media con formato decimal
@printf("Media: %.2f\n", Statistics.mean(ganancia))