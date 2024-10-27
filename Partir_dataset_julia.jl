using Dagger, CSV, DataFrames

# Definir el dataset
@info "Comienza la carga del dataset"
file_path = "/home/joaquintschopp/buckets/b1/datasets/competencia_02_ct.csv.gz"
dataset = CSV.File(file_path; buffer_in_memory=true) |> DataFrame
@info "Carga del dataset completada"

# Definir los meses para el filtro
training_months = [202108, 202107, 202106, 202105, 202104, 202103, 202102, 202101]

# Crear una tarea para filtrar los datos
@info "Filtrando dataset"

# Usar Dagger para crear un flujo de trabajo
filtered_data = Dagger.@apawn dataset[dataset.foto_mes.∈training_months, :]

# Ejecutar el trabajo de Dagger y recoger los resultados
@info "Ejecutando el filtro"
filtered_result = Dagger.collect(filtered_data)

# Guardar el dataset filtrado
output_path = "/home/joaquintschopp/buckets/b1/datasets/competencia_julia_ct.csv.gz"
@info "Guardando el dataset filtrado"
CSV.write(output_path, filtered_result, compress=true)
@info "Dataset guardado con éxito en $(output_path)"

