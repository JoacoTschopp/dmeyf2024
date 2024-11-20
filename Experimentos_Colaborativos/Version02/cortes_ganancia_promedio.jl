using DataFrames, CSV, Statistics

# Ruta del archivo a cargar
input_file_path = "D:\\DATASET-ExperimentoSTACKING-02\\df_merged_completo.csv"

# Cargar el DataFrame desde el archivo CSV
println("Cargando el DataFrame desde el archivo CSV...")
df = CSV.read(input_file_path, DataFrame)


# Función para generar un nuevo DataFrame con las mismas columnas y aplicar el cálculo solicitado
function process_dataframe(df)
    processed_df = DataFrame()

    # Iterar sobre cada columna (excepto las columnas fijas como numero_de_cliente, foto_mes, clase_ternaria)
    for col in names(df)
        if col != "numero_de_cliente" && col != "foto_mes" && col != "clase_ternaria"
            # Ordenar la columna
            sorted_indices = sortperm(df[:, col], rev=true)
            sorted_df = df[sorted_indices, :]

            # Generar cortes desde 8000 a 16000 con un paso de 500
            for corte in 8000:500:16000
                # Filtrar el DataFrame ordenado para realizar el cálculo
                corte_df = sorted_df[1:corte, :]
                baja2_mask = corte_df.clase_ternaria .== "BAJA+2"
                count_baja2 = count(baja2_mask)
                sumatoria = count_baja2 * 280000 - (corte * 7000)

                # Agregar los resultados al nuevo DataFrame
                push!(processed_df, (col_name=col, corte=corte, resultado=sumatoria))
            end
        end
    end

    return processed_df
end

# Procesar el DataFrame y obtener el resultado
println("Procesando el DataFrame...")
result_df = process_dataframe(df)

# Guardar el DataFrame resultante en un nuevo archivo CSV
output_file_path = "D:\\DATASET-ExperimentoSTACKING-02\\processed_df.csv"
println("Guardando el DataFrame procesado en el archivo CSV...")
CSV.write(output_file_path, result_df)

println("Proceso completado. Los resultados se han guardado en: $output_file_path")


# Función para calcular los promedios de cada corte para las columnas que terminan con el mismo sufijo w##
function calculate_averages(result_df)
    grouped_df = groupby(result_df, [:corte])
    suffixes = unique([match(r"w\d{2}$", row[:col_name]).match for row in eachrow(result_df) if match(r"w\d{2}$", row[:col_name]) !== nothing])
    average_df = DataFrame(corte=unique(result_df.corte))

    for suffix in suffixes
        avg_values = Float64[]
        for corte_group in grouped_df
            corte = unique(corte_group.corte)[1]
            valores = [row[:resultado] for row in eachrow(corte_group) if occursin(suffix, row[:col_name])]
            push!(avg_values, isempty(valores) ? NaN : mean(valores))
        end
        average_df[:, Symbol(suffix)] = avg_values
    end

    return average_df
end

# Calcular los promedios para las columnas que terminan en el mismo w##
println("Calculando los promedios para cada corte...")
average_df = calculate_averages(result_df)

# Guardar el DataFrame con los promedios en un nuevo archivo CSV
average_output_file_path = "D:\\DATASET-ExperimentoSTACKING-02\\averages_df.csv"
println("Guardando el DataFrame de promedios en el archivo CSV...")
CSV.write(average_output_file_path, average_df, floatformat="%.2f")

println("Proceso completado. Los resultados de los promedios se han guardado en: $average_output_file_path")
