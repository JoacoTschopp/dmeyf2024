using Dagger, CSV, DataFrames

# Definir el dataset
@info "Comienza la carga del dataset"
file_path = "/home/joaquintschopp/buckets/b1/datasets/competencia_02_ct.csv.gz"
dataset = CSV.File(file_path) |> DataFrame
@info "Carga del dataset completada"

# Definir los meses para el filtro
training_months = [202108, 202107, 202106, 202105, 202104, 202103, 202102, 202101]

# Crear un proceso de filtro paralelo usando Dagger
@info "Filtrando dataset"
filtered_dataset = @spawn dataset[dataset.foto_mes.∈training_months, :]

# Esperar a que se complete la tarea y recoger los datos
filtered_dataset = collect(filtered_dataset)
@info "Filtrado completado"

# Guardar el dataset filtrado
output_path = "/home/joaquintschopp/buckets/b1/datasets/competencia_julia_ct.csv.gz"
@info "Guardando el dataset filtrado"
CSV.write(output_path, filtered_dataset, compress=true)
@info "Dataset guardado con éxito en $(output_path)"
