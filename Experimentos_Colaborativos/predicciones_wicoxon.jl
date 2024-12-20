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
    columnas_m = filter(col -> startswith(col, "m_"), names(df))[1:10]
    columnas_seleccionadas = vcat([:numero_de_cliente,:foto_mes, :clase_ternaria], Symbol.(columnas_m))
    df_seleccionadas = select(df, columnas_seleccionadas)
    filter!(row -> row[:foto_mes] == 202106, df_seleccionadas)


    # Extraer el nivel del archivo del nombre (nivel1_w##)
    m = match(r"nivel1_w(\d{2})", file_name)
    s = match(r"nivel2_s(\d{2})", file_name)
    
    # Asignar el nivel según la coincidencia
    if m !== nothing
        nivel = m.match
    elseif s !== nothing
        nivel = s.match
    else
        nivel = ""
    end

    # Agregar una columna con el promedio de las 5 columnas seleccionadas
    #df_seleccionadas[!, Symbol("prom")] = mean.(eachrow(select(df_seleccionadas, columnas_m)))

    # Renombrar las columnas agregando el nivel extraído, excepto si la columna es "numero_de_cliente"
    renamed_columns = Symbol.([col in ["numero_de_cliente", "foto_mes"] ? col : string(col, "_", nivel) for col in names(df_seleccionadas)])
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
        return outerjoin(df_original, df_nuevo, on=[:numero_de_cliente, :foto_mes], makeunique=true)
    end
end

# Inicializar un DataFrame vacío
global df_total = DataFrame()

# Iterar sobre los archivos y realizar el join
for file in txt_files
    df = generar_dataframe(file)
    global df_total = join_dataframes(df_total, df)
end

# Función para reducir el DataFrame dejando solo una columna foto_mes y clase_ternaria
function reducir_dataframe(df::DataFrame)
    # Seleccionar las columnas que empiezan con "foto_mes"
    foto_mes_cols = filter(col -> startswith(string(col), "foto_mes"), names(df))
    if !isempty(foto_mes_cols)
        df = select(df, Not(foto_mes_cols[2:end]))
        rename!(df, Symbol(foto_mes_cols[1]) => :foto_mes)
    end

    # Seleccionar las columnas que empiezan con "clase_ternaria"
    clase_ternaria_cols = filter(col -> startswith(string(col), "clase_ternaria"), names(df))
    if !isempty(clase_ternaria_cols)
        df = select(df, Not(clase_ternaria_cols[2:end]))
        rename!(df, Symbol(clase_ternaria_cols[1]) => :clase_ternaria)
    end

    return df
end

# Reducir el DataFrame total
df_total = reducir_dataframe(df_total)


# Mostrar el DataFrame combinado
println("Dimensiones del DataFrame combinado: ", size(df_total))

# Guardar el DataFrame combinado en un archivo CSV en la misma dirección
output_file = joinpath(folder_path, "predicciones_wicoxon.csv")
CSV.write(output_file, df_total)




