using CategoricalArrays

using CSV, DataFrames, Random, Statistics
using Primes
using DecisionTree, Impute
using Base.Threads
using Distributions
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

#ptree = ttree(0, 7, 800, 20, 0) #Profesor
min_samples_split = 800
min_samples_leaf = 20

ptree = ttree(0, 7, min_samples_split, min_samples_leaf, 0)
training = 0.7
semilla = 17
qsemillas = 1 #0

### CALCULO GANANCIA

function EstimarGanancia(semilla, training, ptree)
    ganancia_test_normalizada = -1, 0

    # particion
    Random.seed!(semilla)
    vfold = 2 .- Int.(rand(Float64, length(dataset_clase)) .< training)

    # train_campos =  replace!( Matrix( dataset[ vfold .== 1 ,  Not(:clase_ternaria)] ), missing => 0  )

    # clase = string.(dataset[ vfold .== 1 , :clase_ternaria ])
    # datos = Matrix( dataset[ vfold .== 1 ,  Not(:clase_ternaria)] )
    # genero el modelo en training
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

    n_test_subjects = sum(vfold .== 2)
    probabilidades_positivas = pred[:, 2]
    #probabilidades_positivas = [pdf(prob, "BAJA+2") for prob in pred]
    umbral = 1 / 40
    predicciones_baja2 = [prob > umbral ? 1 : 0 for prob in probabilidades_positivas]
    # Contar cuántas predicciones son "BAJA+2"
    n_predicciones_baja2 = sum(predicciones_baja2)
    n_verdaderos_baja2 = sum(dataset_clase[vfold.==2] .== "BAJA+2")
    aciertos_baja2 = sum((dataset_clase[vfold.==2] .== "BAJA+2") .& (predicciones_baja2 .== 1))

    ganancia_test_normalizada = sum(
        (dataset_clase[vfold.==2][(pred[:, 2].>0.025)] .== "BAJA+2")
        .*
        280000
        .-
        7000
    ) / (1.0 - training)

    return ganancia_test_normalizada, n_test_subjects, n_predicciones_baja2, n_verdaderos_baja2, aciertos_baja2
end

####ENTRENAMIENTO Y CALCULO DE MEDIAS

dataset = df
#dataset = sort!(dataset)#, cliente_antiguedad
# restrinjo al periodo 202104
dataset = dataset[dataset.foto_mes.==202104, :]

# Lamentablemente debo imputar nulos, 
#  porque la libreria DecisionTree no los soporta
#Esto seria imputando pro lo que cree mejor segun los datos precentes.
#dataset = Impute.substitute( dataset ) 

#Imputar por 0
dataset = coalesce.(dataset, 0)

# formato para  DecisionTrees
dataset_clase = string.(dataset[:, :clase_ternaria])
dataset_matriz = Matrix(dataset[:, Not(:clase_ternaria)])

# elimino  dataset
dataset = Nothing

# genero la cantidad de qsemillas  nuevas semillas
#####Random.seed!(semilla)
#####semillas = rand(Primes.primes(100000, 999999), qsemillas)
semillas = Vector{Int64}()
push!(semillas, 214363)#115571
push!(semillas, 777349)
push!(semillas, 236917)
push!(semillas, 175621)
push!(semillas, 252277)

# vector donde almaceno los resultados
ganancia = Array{Float64}(undef, length(semillas))


# Crear un DataFrame vacío para almacenar los resultados
resultados = DataFrame(semilla=Int[], ganancia=Float64[], n_test_subjects=Int[], n_predicciones_baja2=Int[], n_verdaderos_baja2=Int[], aciertos_baja2=Int[])

# Calcular las ganancias y almacenar los resultados@threads
@time for i = 1:length(semillas)
    ganancia[i], n_test_subjects, n_predicciones_baja2, n_verdaderos_baja2, aciertos_baja2 = EstimarGanancia(semillas[i], training, ptree)
    println("Semilla: $(semillas[i])")
    println("Ganancia: $(ganancia[i])")
    println("Cantidad de sujetos en testing: $n_test_subjects")
    println("Cantidad de sujetos predichos como BAJA+2: $n_predicciones_baja2")
    println("Cantidad de verdaderos BAJA+2 en testing: $n_verdaderos_baja2")
    println("Cantidad de aciertos (BAJA+2 correctamente predichos): $aciertos_baja2")
    #println("Ganancia: $(ganancia[i])")
    # Agregar los resultados al DataFrame
    #push!(resultados, (semillas[i], ganancia[i], n_test_subjects, n_predicciones_baja2, n_verdaderos_baja2, aciertos_baja2))
end

# Imprimir cada valor en el array con formato decimal
for g in ganancia
    @printf("%.2f\n", g)

end

# Imprimir la media con formato decimal
@printf("Media: %.2f\n", Statistics.mean(ganancia))