using DataFrames
using CSV
using HypothesisTests

# Especificar la ruta del archivo CSV generado previamente
file_path = "G:\\Mi unidad\\01-Maestria Ciencia de Datos\\DMEyF\\Expe_STACKING\\predict\\conjunto_prediciones.csv"

# Definir la columna seleccionada
columna_m_seleccionada = "m_1_9_nivel1_w10"

# Leer el archivo CSV y cargarlo en un DataFrame
df_ganancias = DataFrame(CSV.File(file_path; delim=','))

# Mostrar las dimensiones del DataFrame cargado
println("Dimensiones del DataFrame de ganancias cargado: ", size(df_ganancias))

# Función para realizar el test de Wilcoxon entre la mejor columna y las restantes
function test_wilcoxon(df::DataFrame, columna_mejor::String)
    # Filtrar las columnas de predicción que comienzan con 'm'
    columnas_m = filter(col -> startswith(string(col), "m"), names(df))

    # Eliminar la columna seleccionada de la lista de columnas a comparar
    columnas_m = setdiff(columnas_m, [columna_mejor])

    # Inicializar un DataFrame para almacenar los resultados
    resultados_wilcoxon = DataFrame(columna=String[], p_value=Float64[], estadistico=Float64[])

    # Iterar sobre las columnas y realizar el test de Wilcoxon
    for columna in columnas_m
        # Extraer los pares de valores de las dos columnas a comparar
        x = df[!, Symbol(columna_mejor)]
        y = df[!, Symbol(columna)]

        # Realizar el test de Wilcoxon
        resultado = SignedRankTest(x, y)

        # Almacenar el resultado en el DataFrame de resultados
        push!(resultados_wilcoxon, (columna=columna, p_value=pvalue(resultado), estadistico=resultado.W))
    end

    # Retornar el DataFrame con los resultados
    return resultados_wilcoxon
end

# Llamar a la función para realizar el test de Wilcoxon
df_resultados = test_wilcoxon(df_ganancias, columna_m_seleccionada)

# Mostrar los resultados
println("Resultados del Test de Wilcoxon: ")
println(df_resultados)

# Función para eliminar registros que no validen el p-value
function filtrar_por_pvalue_no_significativo(df::DataFrame, umbral::Float64)
    return filter(row -> row.p_value >= umbral, df)
end

# Aplicar la función para filtrar los resultados por p-value con umbral de 0.05
df_resultados_filtrados = filtrar_por_pvalue_no_significativo(df_resultados, 0.05)

# Mostrar los resultados filtrados
println("Resultados del Test de Wilcoxon que no presentan diferencias significativas (p-value >= 0.05): ")
println(df_resultados_filtrados)

