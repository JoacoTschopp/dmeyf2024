using CSV, DataFrames, Random, Statistics
using Primes
using DecisionTree, Impute
using Base.Threads
using Distributed
using Printf

df = CSV.read("G:/Mi unidad/01-Maestria Ciencia de Datos/DMEyF/TPs/dmeyf-2024/datasets/competencia_01_julia.csv", DataFrame)

# Suponiendo que tu DataFrame se llama `df`
# Agrupar por `foto_mes` y `clase_ternaria`, y contar el número de ocurrencias

grouped = combine(groupby(df, [:foto_mes, :clase_ternaria]), nrow => :count)

# Ordenar el resultado por `foto_mes` para mejor legibilidad
sort!(grouped, [:foto_mes, :clase_ternaria])

# Mostrar el resultado
println(grouped)