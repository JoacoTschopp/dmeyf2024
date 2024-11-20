using DataFrames, CSV, Glob

# Ruta de la carpeta que contiene los archivos
folder_path = "D:\\DATASET-ExperimentoSTACKING-02"

# Obtener una lista de todos los archivos que cumplan con el formato "prediccionfuturo_w##.txt"
file_list = glob("prediccionfuturo_w??.txt", folder_path)
println("Archivos encontrados:")
println(file_list)

# Función para cargar archivos en un DataFrame y agregar el sufijo correspondiente
function load_and_merge_files(file_list)
    merged_df = DataFrame()
    
    # Iterar sobre cada archivo
    for file in file_list
        # Obtener el identificador w## del nombre del archivo
        match_result = match(r"w\d{2}", file)
        suffix = match_result.match
        
        # Cargar el archivo en un DataFrame
        df = CSV.read(file, DataFrame)
        
        # Filtrar solo las columnas que empiezan con 'm'
        m_columns = names(df)[findall(col -> startswith(col, "m"), names(df))][1:20]
        df_m = df[:, m_columns]
        
        # Renombrar las columnas para agregar el sufijo w##
        renamed_columns = Dict(col => "$(col)_$(suffix)" for col in m_columns)
        rename!(df_m, renamed_columns)
        
        # Unir los datos por "numero_de_cliente", "foto_mes" y agregar "clase_ternaria"
        if nrow(merged_df) == 0
            merged_df = hcat(df[:, ["numero_de_cliente", "foto_mes", "clase_ternaria"]], df_m)
        else
            merged_df = outerjoin(merged_df, hcat(df[:, ["numero_de_cliente", "foto_mes", "clase_ternaria"]], df_m), on = ["numero_de_cliente", "foto_mes", "clase_ternaria"], makeunique=true)
        end
    end
    
    return merged_df
end

# Función para realizar un nuevo merge con solo las primeras 5 columnas que empiezan con 'm'
function load_and_merge_first_5_m_columns(file_list)
    merged_df = DataFrame()
    
    # Iterar sobre cada archivo
    for file in file_list
        # Obtener el identificador w## del nombre del archivo
        match_result = match(r"w\d{2}", file)
        suffix = match_result.match
        
        # Cargar el archivo en un DataFrame
        df = CSV.read(file, DataFrame)
        
        # Filtrar solo las primeras 5 columnas que empiezan con 'm'
        m_columns = names(df)[findall(col -> startswith(col, "m"), names(df))][1:5]
        df_m = df[:, m_columns]
        
        # Renombrar las columnas para agregar el sufijo w##
        renamed_columns = Dict(col => "$(col)_$(suffix)" for col in m_columns)
        rename!(df_m, renamed_columns)
        
        # Unir los datos por "numero_de_cliente", "foto_mes" y agregar "clase_ternaria"
        if nrow(merged_df) == 0
            merged_df = hcat(df[:, ["numero_de_cliente", "foto_mes", "clase_ternaria"]], df_m)
        else
            merged_df = outerjoin(merged_df, hcat(df[:, ["numero_de_cliente", "foto_mes", "clase_ternaria"]], df_m), on = ["numero_de_cliente", "foto_mes", "clase_ternaria"], makeunique=true)
        end
    end
    
    return merged_df
end

# Llamar a la función para cargar y combinar los archivos
println("DataFrame combinado con todas las columnas que empiezan con 'm':")
df_merged = load_and_merge_files(file_list)
println(size(df_merged))

# Guardar el DataFrame combinado en un archivo CSV
CSV.write("D:\\DATASET-ExperimentoSTACKING-02\\df_merged_completo.csv", df_merged)

# Llamar a la función para cargar y combinar solo las primeras 5 columnas que empiezan con 'm'
println("DataFrame combinado con las primeras 5 columnas que empiezan con 'm':")
df_merged_first_5 = load_and_merge_first_5_m_columns(file_list)
println(size(df_merged_first_5))

# Guardar el DataFrame combinado con las primeras 5 columnas en un archivo CSV
CSV.write("D:\\DATASET-ExperimentoSTACKING-02\\df_merged_first_5.csv", df_merged_first_5)
