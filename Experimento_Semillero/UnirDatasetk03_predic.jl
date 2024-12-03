using DataFrames, CSV, Glob

# Ruta de la carpeta que contiene los archivos TXT
folder_path = "G:\\Mi unidad\\01-Maestria Ciencia de Datos\\DMEyF\\TENDENCIA Kaggle03"

# Obtener la lista de todos los archivos TXT en la carpeta
file_list = glob("*.txt", folder_path)
println("Archivos encontrados:")
println(file_list)

# Concatenar los archivos en un nuevo DataFrame
global combined_df = DataFrame()
for (i, file) in enumerate(file_list)
    # Cargar el archivo TXT separado por tabuladores
    temp_df = CSV.read(file, DataFrame; delim='\t')

    # Filtrar solo las columnas requeridas
    temp_df = temp_df[:, ["numero_de_cliente", "foto_mes", "Predicted"]]

    # Concatenar al DataFrame combinado
    if i == 1
        global combined_df = temp_df
    else
        global combined_df = vcat(combined_df, temp_df, cols=:union)
    end
end

# Guardar el DataFrame combinado en un nuevo archivo CSV
output_file_path = "G:\\Mi unidad\\01-Maestria Ciencia de Datos\\DMEyF\\TENDENCIA Kaggle03\\Predicted.csv"
println("Guardando el DataFrame combinado en el archivo CSV...")
CSV.write(output_file_path, combined_df, floatformat="%.5f")

println("Proceso completado. El DataFrame combinado se ha guardado en: $output_file_path")
