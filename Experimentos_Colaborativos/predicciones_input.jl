using Glob
using FilePaths
using DataFrames
using Statistics
using CSV

# Especificar la ruta de la carpeta
folder_path = "G:\\Mi unidad\\01-Maestria Ciencia de Datos\\DMEyF\\Expe_STACKING\\predict"

# Obtener la lista de archivos .txt en la carpeta
txt_files = glob("*.txt", folder_path)

# Mostrar los archivos encontrados
for file in txt_files
    println(file)
end

# Función para generar un DataFrame a partir de un archivo
function generar_dataframe(file_name::String)
    # Leer el archivo con delimitador de tabulación y con encabezado
    df = DataFrame(CSV.File(file_name; delim='	', header=true))

    # Seleccionar solo las primeras 5 columnas que comienzan con 'm_'
    columnas_m = filter(col -> startswith(col, "m_"), names(df))[1:5]
    columnas_seleccionadas = vcat([:numero_de_cliente], Symbol.(columnas_m))
    df_seleccionadas = select(df, columnas_seleccionadas)

    # Extraer el nivel del archivo del nombre (nivel1_w##)
    m = match(r"nivel1_w(\d{2})", file_name)
    nivel = m !== nothing ? m.match : ""

    # Agregar una columna con el promedio de las 5 columnas seleccionadas
    df_seleccionadas[!, Symbol("prom")] = mean.(eachrow(select(df_seleccionadas, columnas_m)))

    # Renombrar las columnas agregando el nivel extraído, excepto si la columna es "numero_de_cliente"
    renamed_columns = Symbol.([col == "numero_de_cliente" ? col : string(col, "_", nivel) for col in names(df_seleccionadas)])
    rename!(df_seleccionadas, renamed_columns)

    # Mostrar las dimensiones del DataFrame
    println("Dimensiones del DataFrame: ", size(df))

    # Agregar una columna con el promedio de las 5 columnas seleccionadas


    # Retornar el DataFrame
    return df_seleccionadas
end

# Función para realizar join de DataFrames
function join_dataframes(df_original::DataFrame, df_nuevo::DataFrame)
    if isempty(df_original)
        return df_nuevo
    else
        return outerjoin(df_original, df_nuevo, on=:numero_de_cliente, makeunique=true)
    end
end

# Inicializar un DataFrame vacío
global df_total = DataFrame()

# Iterar sobre los archivos y realizar el join
for file in txt_files
    df = generar_dataframe(file)
    global df_total = join_dataframes(df_total, df)
end

# Mostrar el DataFrame combinado
println("Dimensiones del DataFrame combinado: ", size(df_total))

# Guardar el DataFrame combinado en un archivo CSV en la misma dirección
output_file = joinpath(folder_path, "predicciones_input.csv")
CSV.write(output_file, df_total)




