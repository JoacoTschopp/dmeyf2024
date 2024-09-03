using CSV
using DataFrames
using Statistics
using Printf
using Test
using HypothesisTests
using Base.Threads

using Distributed

# Cargar el archivo CSV en un DataFrame
#file_path = "G:/Mi unidad/01-Maestria Ciencia de Datos/DMEyF/TPs/dmeyf-2024/dmeyf2024/HiperparametrosWilcox/resultados-001.csv"

# Cargar el archivo CSV en un DataFrame
file_path = "G:/Mi unidad/01-Maestria Ciencia de Datos/DMEyF/TPs/dmeyf-2024/dmeyf2024/HiperparametrosWilcox/resultados-002.csv"
df = CSV.read(file_path, DataFrame)

# Agrupar los datos por los hiperparámetros
grouped_df = groupby(df, [:min_samples_split, :maxdepth, :n_subfeatures, :min_samples_leaf, :min_purity_increase])

# Crear un nuevo DataFrame que contenga los hiperparámetros y las 10 mediciones de ganancia
new_df = DataFrame(
    min_samples_split = Int[],
    maxdepth = Int[],
    n_subfeatures = Int[],
    min_samples_leaf = Int[],
    min_purity_increase = Float64[],
    ganancias = Array{Float64,1}[],  # Columna para almacenar las mediciones de ganancia como un arreglo
    promedio_ganancias = Float64[]   # Nueva columna para almacenar el promedio de las ganancias
)

# Llenar el nuevo DataFrame
for g in grouped_df
    ganancias = g.ganancia
    push!(new_df, (
        g.min_samples_split[1],
        g.maxdepth[1],
        g.n_subfeatures[1],
        g.min_samples_leaf[1],
        g.min_purity_increase[1],
        ganancias,
        mean(ganancias)  # Calcular el promedio de las ganancias
    ))
end


# Función para mostrar hiperparámetros donde la media de ganancias es 0.0
function show_params_mean_zero(df::DataFrame)
    filtered_df = filter(row -> row.promedio_ganancias == 0.0, df)
    println("Hiperparámetros con media de ganancias igual a 0.0:")
    println(filtered_df)
end

# Función para mostrar hiperparámetros donde al menos una ganancia es 0.0, pero no todas
function show_params_some_zero(df::DataFrame)
    filtered_df = filter(row -> any(x -> x == 0.0, row.ganancias) && !all(x -> x == 0.0, row.ganancias), df)
    println("Hiperparámetros con al menos una ganancia igual a 0.0, pero no todas:")
    println(filtered_df)
end

# Función para mostrar los hiperparámetros con la ganancia promedio mayor
function show_params_max_mean(df::DataFrame)
    max_mean = maximum(df.promedio_ganancias)
    filtered_df = filter(row -> row.promedio_ganancias == max_mean, df)
    @printf("Hiperparámetros con la ganancia promedio mayor (%.4f):\n", max_mean)
    println(filtered_df)
end

# Mostrar los resultados usando las funciones definidas
show_params_mean_zero(new_df)
show_params_some_zero(new_df)
show_params_max_mean(new_df)

# Filtrar los registros que no contienen ningún 0.0 en el array de ganancias
df = filter(row -> !any(x -> x == 0.0, row.ganancias), new_df)

# Función para calcular el test de Wilcoxon entre dos arrays de ganancias
function wilcoxon_test(arr1::Vector{Float64}, arr2::Vector{Float64})
    test_result = MannWhitneyUTest(arr1, arr2)  # Test de Mann-Whitney (equivalente al test de Wilcoxon)
    return pvalue(test_result) > 0.05  # Retorna true si el test es significativo con un p-valor > 0.05
end

# Función para filtrar arrays en el DataFrame usando multihilo
function filter_by_wilcoxon(df::DataFrame)
    # Ordenar el DataFrame por ganancia promedio
    df = sort(df, :promedio_ganancias, rev=true)

    i = 1
    while i <= length(df.ganancias)
        j = i + 1
        while j <= length(df.ganancias)
            if wilcoxon_test(df.ganancias[i], df.ganancias[j])
                # Eliminar el registro j si el test no es significativo
                delete!(df, j)
            else
                j += 1  # Incrementar j solo si no se elimina el registro
            end
        end
        i += 1
    end

    return df
end

print("Antes del test exiten tantos registros: ")
println(size(df,1))

df_hiperP = filter_by_wilcoxon(df)

println(df_hiperP)
