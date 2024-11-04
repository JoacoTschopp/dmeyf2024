using DataFrames
using CSV

# Especificar la ruta del archivo CSV generado previamente
file_path = "G:\\Mi unidad\\01-Maestria Ciencia de Datos\\DMEyF\\Expe_STACKING\\predict\\conjunto_prediciones.csv"

# Definir la variable corte
corte = 11000

# Leer el archivo CSV y cargarlo en un DataFrame
df = DataFrame(CSV.File(file_path))

# Mostrar las dimensiones del DataFrame cargado
println("Dimensiones del DataFrame cargado: ", size(df))

# Crear un nuevo DataFrame con numero_de_cliente, clase_ternaria y una columna que empieza con 'm'
function generar_sub_dataframe(df::DataFrame, columna_m::String)
    return select(df, :numero_de_cliente, :clase_ternaria, Symbol(columna_m))
end

# Recorrer los nombres de las columnas que empiezan con 'm'
m_columnas = filter(col -> startswith(string(col), "m"), names(df))

# Inicializar un DataFrame para almacenar las ganancias
df_ganancias = DataFrame(columna_m=String[], ganancia=Int64[])

# Función para procesar el DataFrame
function procesar_dataframe(sub_df::DataFrame, numero::Int)
    println("Procesando DataFrame con número: ", numero)

    # Ordenar el DataFrame de mayor a menor según la columna 'm'
    columna_m = names(sub_df)[3]  # Obtener el nombre de la columna que empieza con 'm'
    sort!(sub_df, Symbol(columna_m), rev=true)

    # Seleccionar los primeros 'numero' registros
    top_df = first(sub_df, numero)

    # Contar cuántos tienen clase_ternaria == "BAJA+2"
    count_baja2 = count(row -> row[:clase_ternaria] == "BAJA+2", eachrow(top_df))

    # Calcular el resultado: (count_baja2 * 280000) - (numero * 7000)
    resultado = Int64((count_baja2 * 280) - (numero * 7))

    # Retornar el resultado
    return resultado
end

# Generar un nuevo DataFrame para cada columna que empieza con 'm' y llamar a una función
for columna_m in m_columnas
    sub_df = generar_sub_dataframe(df, columna_m)
    println("DataFrame generado con la columna: ", columna_m)

    # Llamar a la función que pasa el sub_df junto con el número 11000
    ganancia = procesar_dataframe(sub_df, corte)

    # Agregar la ganancia al DataFrame de ganancias
    push!(df_ganancias, (columna_m=columna_m, ganancia=ganancia))
end

# Guardar el DataFrame de ganancias en un archivo CSV
output_file = "G:\\Mi unidad\\01-Maestria Ciencia de Datos\\DMEyF\\Expe_STACKING\\predict\\Ganancias_$corte.csv"
CSV.write(output_file, df_ganancias; delim=';')
