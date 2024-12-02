using DataFrames
using CSV
using HypothesisTests


# Definir la ruta del archivo
file_path = "C:\\Users\\tschoppj\\Downloads\\202008-Mejorcorte10500.txt"

# Leer el archivo .txt separado por tabulaciones
df = CSV.File(file_path, delim='\t') |> DataFrame

# Mostrar los nombres de las columnas
println("Column names: ", names(df))

# Mostrar la dimensión del DataFrame
println("DataFrame dimensions: ", size(df))

# Filtrar las columnas que comienzan con 'sem'
sem_columns = filter(col -> startswith(col, "sem"), names(df))

# Ordenar de mayor a menor por la columna 'sem_1_4'
df_sorted = sort(df, :sem_1_4, rev=true)

# Tomar los primeros 10500 registros
subset_df = df_sorted[1:10500, :]

# Comparar 'sem_1_4' contra el resto de las columnas que comienzan con 'sem'
sem_1_4_values = subset_df[:, :sem_1_4]

for col in sem_columns
    if col != "sem_1_4"
        other_values = subset_df[:, col]

        if length(sem_1_4_values) == length(other_values) && length(sem_1_4_values) > 0
            test_result = SignedRankTest(sem_1_4_values, other_values)
            p_value = pvalue(test_result)

            if p_value < 0.05
                println("Wilcoxon Signed-Rank test result between 'sem_1_4' and '", col, "': p-value = ", p_value)

                # Informar los clientes que son realmente diferentes
                clientes_diferentes = subset_df.numero_de_cliente[sem_1_4_values.!=other_values]
                println("Clientes diferentes entre 'sem_1_4' y '", col, "': ", clientes_diferentes)
            end
        end
    end
end