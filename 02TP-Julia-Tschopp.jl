using CSV, DataFrames, Random, Statistics

df = CSV.read("G:/Mi unidad/01-Maestria Ciencia de Datos/DMEyF/TPs/dmeyf-2024/datasets/competencia_01_julia.csv", DataFrame)

#print(names(df))

println(length(unique(df.numero_de_cliente)))

# Función para contar cuántos registros hay de cada valor único en una columna
function contar_valores_unicos(df::DataFrame, columna::Symbol)
    # Agrupar por la columna y contar cuántos hay en cada grupo
    conteo = combine(groupby(df, columna), nrow => :Cantidad)
    
    # Imprimir los resultados
    println("Conteo de valores únicos en la columna $columna:")
    println(conteo)
    
    return conteo
end

function verificar_missing(df::DataFrame)
    # Filtrar filas donde 'clase_ternaria' es missing y 'foto_mes' es 202105 o 202106
    missing_rows = filter(row -> ismissing(row.clase_ternaria) && (row.foto_mes == 202105 || row.foto_mes == 202106), df)
    
    # Mostrar cuántos valores missing hay para esos meses
    println("Cantidad de registros con 'clase_ternaria' missing para 'foto_mes' 202105 o 202106: ", nrow(missing_rows))
    
    return missing_rows
end

# Función para verificar y reemplazar los missing en 'clase_ternaria'
function verificar_y_completar_missing(df::DataFrame)
    # Filtrar filas donde 'clase_ternaria' es missing y 'foto_mes' es 202105 o 202106
    missing_rows = filter(row -> ismissing(row.clase_ternaria) && (row.foto_mes == 202105 || row.foto_mes == 202106), df)
    
    # Mostrar cuántos valores missing hay para esos meses
    println("Cantidad de registros con 'clase_ternaria' missing para 'foto_mes' 202105 o 202106: ", nrow(missing_rows))
    
    # Reemplazar los valores missing en la columna 'clase_ternaria' por " "
    df.clase_ternaria = coalesce.(df.clase_ternaria, " ")
    
    # Verificar que se han reemplazado los valores missing
    println("Missing values reemplazados por ' ' en la columna 'clase_ternaria'.")
    
    return df
end



# Definir la función Filtrado_baja_1
function Filtrado_baja_2(df::DataFrame)
    # Filtrar por 'clase_ternaria' == "BAJA+2"
    df_baja = filter(row -> row.clase_ternaria == "BAJA+2", df)
    
    # Extraer los 'numero_de_cliente' únicos
    clientes_baja = unique(df_baja.numero_de_cliente)
    
    # Generar un nuevo DataFrame con todos los registros de esos clientes
    df_filtrado = filter(row -> row.numero_de_cliente in clientes_baja, df)
    
    # Mostrar la cantidad de clientes con 'BAJA+2'
    println("Cantidad de clientes con BAJA+2: ", length(clientes_baja))
    
    # Mostrar la cantidad de registros finales en el nuevo DataFrame
    println("Cantidad de registros finales: ", nrow(df_filtrado))
    
    return df_filtrado
end

#Verificacion y preparado
val_unicos =  contar_valores_unicos(df, :clase_ternaria)
missing_rows = verificar_missing(df)
df_completado = verificar_y_completar_missing(df)

#Filtrado
df_filtrado = Filtrado_baja_2(df_completado)