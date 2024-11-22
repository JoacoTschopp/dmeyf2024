using DataFrames, CSV

# Ruta del archivo a cargar
input_file_path = "D:\\DATASET-ExperimentoSTACKING-02\\expw227_SC-0005_tb_future_prediccion.txt"

# Cargar el archivo separado por tabuladores
println("Cargando el archivo de predicciones...")
df = CSV.read(input_file_path, DataFrame; delim='\t')

# Filtrar las columnas requeridas
println("Filtrando las columnas seleccionadas...")
columns_to_keep = ["numero_de_cliente", "foto_mes", "clase_ternaria", "sem_1_1", "sem_1_2", "sem_1_3", "sem_1_4", "sem_1_5", "sem_1_6", "sem_1_7", "sem_1_8", "sem_1_9", "sem_1_10"]
filtered_df = df[:, columns_to_keep]

# Guardar el DataFrame filtrado en un nuevo archivo CSV
output_file_path = "D:\\DATASET-ExperimentoSTACKING-02\\predicciones_Semillero_s10.csv"
println("Guardando el DataFrame filtrado en el archivo CSV...")
CSV.write(output_file_path, filtered_df)

println("Proceso completado. El archivo filtrado se ha guardado en: $output_file_path")



# Generar cortes para cada columna que empieza con 'sem'
println("Generando cortes para cada columna que empieza con 'sem'...")

# Crear el directorio para guardar los archivos de los cortes
cortes_folder_path = "D:\\DATASET-ExperimentoSTACKING-02\\Cortes"
mkdir(cortes_folder_path)

# Iterar sobre cada columna que empieza con 'sem'
for col in names(filtered_df)
    if startswith(col, "sem")
        for corte in 9000:500:13500
            # Ordenar la columna de mayor a menor
            sorted_indices = sortperm(filtered_df[:, col], rev=true)
            sorted_df = filtered_df[sorted_indices, :]

            # Crear una columna Predicted con 1 para los primeros 'corte' registros y 0 para el resto
            predicted = [i <= corte ? 1 : 0 for i in 1:nrow(sorted_df)]
            corte_df = DataFrame(numero_de_cliente=sorted_df.numero_de_cliente, Predicted=predicted)

            # Guardar el archivo CSV del corte
            corte_output_path = "$(cortes_folder_path)\\Semillero20_2_$(col)_corte$(corte).csv"
            println("Guardando el archivo del corte en: $corte_output_path")
            CSV.write(corte_output_path, corte_df, floatformat="%.5f")
        end
    end
end

println("Proceso completado. Los archivos de los cortes se han guardado en la carpeta: $cortes_folder_path")
