using DataFrames, CSV
using Statistics

# Ruta del archivo a cargar
input_file_path = "c:\\tb_future_prediccion.txt"

# Cargar el archivo separado por tabuladores
println("Cargando el archivo de predicciones...")
df = CSV.read(input_file_path, DataFrame; delim='\t')

# Mostrar las dimensiones del DataFrame
println("Dimensiones del DataFrame: ", size(df))

# Generar una columna por cada columna que empieza con 'm' que contenga un 1 para los 11000 registros más altos
println("Generando las columnas '_predict'...")
m_columns = filter(col_name -> startswith(col_name, "m"), names(df))

for col in m_columns
    # Ordenar la columna de mayor a menor
    sorted_indices = sortperm(df[!, col], rev=true)
    predicted_values = zeros(Int, nrow(df))
    # Asignar 1 a los 11000 registros más altos
    for i in 1:min(11000, nrow(df))
        predicted_values[sorted_indices[i]] = 1
    end
    # Crear la nueva columna con el sufijo '_predict'
    df[!, Symbol(col * "_predict")] = predicted_values
end

# Generar una columna con la suma de todas las columnas '_predict'
println("Generando la columna 'Predicted_sum'...")
predict_columns = filter(col_name -> endswith(col_name, "_predict"), names(df))
df[!, :Predicted_sum] = [sum(row) for row in eachrow(df[:, predict_columns])]

# Generar un nuevo DataFrame con solo 'numero_de_cliente' y 'Predicted_sum'
println("Generando el nuevo DataFrame con 'numero_de_cliente' y 'Predicted_sum'...")
new_df = df[:, ["numero_de_cliente", "Predicted_sum"]]

# Guardar el DataFrame en un archivo CSV
output_file_path = "c:\\Users\\tschoppj\\Downloads\\submitk03\\predicciones_suma.csv"
println("Guardando el DataFrame en el archivo CSV...")
CSV.write(output_file_path, new_df, floatformat="%.5f")

println("Proceso completado. El archivo se ha guardado en: $output_file_path")

# Generar cortes para Kaggle
println("Generando cortes para Kaggle...")

# Realizar cortes basados en 'Predicted_sum'
for corte in 9000:500:12000
    # Asignar en la columna Predicted 1 a los que tienen más de 60 votos en 'Predicted_sum'
    predicted_values = [value > 60 ? 1 : 0 for value in new_df.Predicted_sum]
    corte_df = DataFrame(numero_de_cliente=new_df.numero_de_cliente, Predicted=predicted_values)

    # Guardar el archivo CSV del corte
    corte_output_path = "c:\\Users\\tschoppj\\Downloads\\submitk03\\VOTING100_$corte.csv"
    println("Guardando el archivo del corte en: $corte_output_path")
    CSV.write(corte_output_path, corte_df, floatformat="%.5f")
end

println("Proceso completado. Los archivos de los cortes se han guardado en la carpeta de salida.")

