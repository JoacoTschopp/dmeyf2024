using JuliaDB, CSV, DataFrames, Dagger

# Definir el dataset y cargar datos en JuliaDB
@info "Comienza carga de Dataset"
file_path = "/home/joaquintschopp/buckets/b1/datasets/competencia_02_ct.csv.gz"
output_path = "/home/joaquintschopp/buckets/b1/datasets/competencia_julia_ct.csv.gz"

# Cargar el archivo con DataFrames para luego convertirlo
data_df = DataFrame(CSV.File(file_path; buffer_in_memory=true))
@info "Fin de la carga inicial en DataFrame"

# Crear una tabla de JuliaDB desde el DataFrame
dataset = table(data_df)

# Definir los valores de `foto_mes` a mantener
foto_mes_filtrar = [202108, 202107, 202106, 202105, 202104, 202103, 202102, 202101]

# Filtrar los datos con JuliaDB
@info "Comienza filtrado con JuliaDB"
filtered_data = filter(row -> row.foto_mes in foto_mes_filtrar, dataset)
@info "Fin del filtrado con JuliaDB"

# Convertir la tabla filtrada en un DataFrame para guardar
filtered_df = DataFrame(filtered_data)

# Guardar el DataFrame filtrado como CSV comprimido
@info "Guardando el dataset filtrado en CSV comprimido"
CSV.write(output_path, filtered_df; gzip=true)
@info "Proceso completo: dataset guardado en $(output_path)"
