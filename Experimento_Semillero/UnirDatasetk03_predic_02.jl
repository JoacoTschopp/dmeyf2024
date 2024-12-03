using DataFrames, CSV

# Rutas de los archivos
predicted_file_path = "/home/joaquintschopp/buckets/b1/datasets/Predicted.csv"
competencia_file_path = "/home/joaquintschopp/buckets/b1/datasets/competencia_03_ct.csv.gz"

# Cargar los DataFrames desde los archivos
println("Cargando el archivo Predicted.csv...")
predicted_df = CSV.read(predicted_file_path, DataFrame)

println("Cargando el archivo competencia_03_ct.csv.gz...")
competencia_df = CSV.read(competencia_file_path, DataFrame)

# Mostrar las dimensiones de los DataFrames
println("Dimensiones del DataFrame Predicted: ", size(predicted_df))
println("Dimensiones del DataFrame Competencia: ", size(competencia_df))

# Unir los DataFrames teniendo en cuenta 'numero_de_cliente' y 'foto_mes'
println("Uniendo los DataFrames...")
combined_df = outerjoin(competencia_df, predicted_df, on=["numero_de_cliente", "foto_mes"], makeunique=true)

# Guardar el DataFrame combinado en un nuevo archivo CSV comprimido
output_file_path = "/home/tschopp_joaquin333/buckets/b1/datasets/competencia_04_ct.csv.gz"
println("Guardando el DataFrame combinado en el archivo CSV comprimido...")
CSV.write(output_file_path, combined_df, floatformat="%.5f")

println("Proceso completado. El DataFrame combinado se ha guardado en: $output_file_path")
