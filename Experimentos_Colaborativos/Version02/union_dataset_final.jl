using DataFrames, CSV, Glob

# Ruta del archivo con las primeras 5 columnas
first_5_columns_path = "D:\\DATASET-ExperimentoSTACKING-02\\df_merged_first_5.csv"

# Cargar el DataFrame con las primeras 5 columnas
println("Cargando el DataFrame con las primeras 5 columnas...")
df_merged_first_5 = CSV.read(first_5_columns_path, DataFrame)

# Ruta de la carpeta que contiene los archivos con el formato "prediccion_######.txt"
folder_path = "D:\\DATASET-ExperimentoSTACKING-02"
file_list = glob("prediccion_*.txt", folder_path)
println("Archivos encontrados:")
println(file_list)

# Concatenar los archivos en un nuevo DataFrame
concatenated_df = DataFrame()
for (i, file) in enumerate(file_list)
    global concatenated_df
    temp_df = CSV.read(file, DataFrame)
    rename!(temp_df, names(df_merged_first_5))
    if i == 1
        concatenated_df = temp_df
    else
        concatenated_df = vcat(concatenated_df, temp_df, cols=:union)
    end
end

# Concatenar los registros del df_merged_first_5 debajo de concatenated_df
println("Concatenando DataFrames...")
combined_df = vcat(df_merged_first_5, concatenated_df, cols=:union)

# Guardar el DataFrame combinado en un archivo CSV
output_file_path = "D:\\DATASET-ExperimentoSTACKING-02\\dataset_stacking02_S1.csv"
println("Guardando el DataFrame combinado en el archivo CSV...")
CSV.write(output_file_path, combined_df, floatformat="%.2f")

println("Proceso completado. El DataFrame combinado se ha guardado en: $output_file_path")


# Cargar el archivo competencia_02_ct.csv
competencia_file_path = "D:\\DATASET-ExperimentoSTACKING-02\\competencia_02_ct.csv"
println("Cargando el archivo competencia_02_ct.csv...")
competencia_df = CSV.read(competencia_file_path, DataFrame)

# Filtrar el nuevo DataFrame para incluir solo los foto_mes de combined_df
println("Filtrando el DataFrame competencia para incluir solo los foto_mes de combined_df...")
filtered_competencia_df = filter(row -> row.foto_mes in combined_df.foto_mes, competencia_df)

# Concatenar combined_df con el DataFrame competencia filtrado tomando como clave numero_de_cliente y foto_mes
println("Concatenando el DataFrame combinado con el DataFrame competencia filtrado...")
final_combined_df = outerjoin(combined_df, filtered_competencia_df, on=["numero_de_cliente", "foto_mes"], makeunique=true)

# Guardar el DataFrame final combinado en un archivo CSV
final_output_file_path = "D:\\DATASET-ExperimentoSTACKING-02\\dataset_stacking02_S2.csv"
println("Guardando el DataFrame final combinado en el archivo CSV...")
CSV.write(final_output_file_path, final_combined_df, floatformat="%.5f")

println("Proceso completado. El DataFrame final combinado se ha guardado en: $final_output_file_path")
