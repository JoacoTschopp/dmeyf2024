using DataFrames, CSV
using Statistics

# Ruta del archivo a cargar
input_file_path = "c:\\tb_future_prediccion.txt"

# Cargar el archivo separado por tabuladores
println("Cargando el archivo de predicciones...")
df = CSV.read(input_file_path, DataFrame; delim='\t')

# Mostrar las dimensiones del DataFrame
println("Dimensiones del DataFrame: ", size(df))

# Generar un nuevo atributo con el promedio de las columnas que empiezan con 'm'
println("Generando el atributo Predicted...")
m_columns = filter(col_name -> startswith(col_name, "m"), names(df))
df[!, :Predicted] = [mean(row) for row in eachrow(df[:, m_columns])]

# Generar un nuevo DataFrame con solo 'numero_de_cliente' y 'Predicted'
println("Generando el nuevo DataFrame con 'numero_de_cliente' y 'Predicted'...")
new_df = df[:, ["numero_de_cliente", "Predicted"]]

# Guardar el DataFrame en un archivo CSV
output_file_path = "c:\\Users\\tschoppj\\Downloads\\submitk03\\predicciones_promedio.csv"
println("Guardando el DataFrame en el archivo CSV...")
CSV.write(output_file_path, new_df, floatformat="%.5f")

println("Proceso completado. El archivo se ha guardado en: $output_file_path")

# Generar cortes para Kaggle
println("Generando cortes para Kaggle...")

# Ordenar las predicciones de 'Predicted' de mayor a menor
sorted_df = sort(new_df, :Predicted, rev=true)

# Realizar cortes desde 9000 a 12000 y generar archivos para cada corte
for corte in 9000:500:12000
    # Asignar en la columna Predicted 1 a los que están dentro del corte y 0 al resto
    predicted_values = [i <= corte ? 1 : 0 for i in 1:nrow(sorted_df)]
    corte_df = DataFrame(numero_de_cliente=sorted_df.numero_de_cliente, Predicted=predicted_values)

    # Guardar el archivo CSV del corte
    corte_output_path = "c:\\Users\\tschoppj\\Downloads\\submitk03\\EMBEDING100_$corte.csv"
    println("Guardando el archivo del corte en: $corte_output_path")
    CSV.write(corte_output_path, corte_df, floatformat="%.5f")
end

println("Proceso completado. Los archivos de los cortes se han guardado en la carpeta de salida.")
