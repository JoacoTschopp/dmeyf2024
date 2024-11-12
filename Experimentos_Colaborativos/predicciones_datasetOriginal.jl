# Cargar las librerías necesarias
using DataFrames
using CSV

# Definir las rutas de los archivos de los datasets
path_df_predict = "G:\\Mi unidad\\01-Maestria Ciencia de Datos\\DMEyF\\Expe_STACKING\\predict\\predicciones_input.csv"
path_df_kaggle02 = "D:\\DmEyF_Julia\\dataset\\competencia_02_ct.csv.gz"

##################################################################################

# Cargar el dataset df_kaggle02
println("Cargando df_kaggle02 desde: ", path_df_kaggle02)
df_kaggle02 = DataFrame(CSV.File(path_df_kaggle02))

# Mostrar las dimensiones del dataset df_kaggle02 después de filtrar
println("Dimensiones de df_kaggle02 antes de filtrar por 'foto_mes': ", size(df_kaggle02))

# Filtrar los registros del dataset df_kaggle02 donde "foto_mes" esté en el rango específico
fechas_filtrar = [202007, 202008, 202009, 202010, 202011, 202012, 202101, 202102, 202103, 202104, 202105, 202106]
df_kaggle02_filtrado = filter(row -> row.foto_mes in fechas_filtrar, df_kaggle02)

# Mostrar las dimensiones del dataset df_kaggle02 después de filtrar
println("Dimensiones de df_kaggle02 después de filtrar por 'foto_mes': ", size(df_kaggle02_filtrado))

##################################################################################

# Cargar el dataset df_predict
println("Cargando df_predict desde: ", path_df_predict)
df_predict = DataFrame(CSV.File(path_df_predict))

# Mostrar las dimensiones del dataset df_predict después de quitar la columna
println("Dimensiones de df_predict: ", size(df_predict))

# Quitar las columnas "foto_mes" y "clase_ternaria" del dataset df_predict
select!(df_predict, Not([:foto_mes, :clase_ternaria]))

# Mostrar las dimensiones del dataset df_predict después de quitar las columnas
println("Dimensiones de df_predict después de quitar 'foto_mes' y 'clase_ternaria': ", size(df_predict))

#################################################################################

# Realizar un merge de df_predict y df_kaggle02_filtrado
println("Realizando el merge de df_predict y df_kaggle02_filtrado")
df_experimento = innerjoin(df_kaggle02_filtrado, df_predict, on=:numero_de_cliente)

# Mostrar las dimensiones del dataset combinado
dimension_experimento = size(df_experimento)
println("Dimensiones del dataset combinado df_experimento: ", dimension_experimento)

# Guardar el DataFrame combinado en un archivo CSV comprimido en la carpeta especificada
output_path = "G:\\Mi unidad\\01-Maestria Ciencia de Datos\\DMEyF\\Expe_STACKING\\predict\\experimento_stacking_02.csv.gz"
println("Guardando df_experimento en: ", output_path)
CSV.write(output_path, df_experimento)