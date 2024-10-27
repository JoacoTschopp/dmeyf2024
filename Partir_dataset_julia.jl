using Dagger, CSV, DataFrames

# Ruta de archivo y parámetros de filtro
file_path = "/home/joaquintschopp/buckets/b1/datasets/competencia_02_ct.csv.gz"
output_path = "/home/joaquintschopp/buckets/b1/datasets/competencia_julia_ct.csv"
training_months = Set([202108, 202107, 202106, 202105, 202104, 202103, 202102, 202101])  # Convertido a Set para mejorar eficiencia

# Cargar el dataset completo
@info "Comienza carga del dataset"
dataset = DataFrame(CSV.File(file_path; buffer_in_memory=true))

# Definir una función de filtro para trabajar con `Dagger`
function filter_training_months(df, months)
    return filter(row -> row.foto_mes in months, df)
end

# Ejecutar el filtro en paralelo
@info "Inicia filtrado paralelo con Dagger"
filtered_data_future = Dagger.@spawn filter_training_months(dataset, training_months)
filtered_data = fetch(filtered_data_future)

# Guardar el resultado
@info "Guardando el dataset filtrado"
CSV.write(output_path, filtered_data; compress=true)
@info "Proceso completo, archivo guardado en $output_path"


