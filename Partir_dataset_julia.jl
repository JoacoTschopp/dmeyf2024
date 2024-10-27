using Dagger, CSV, DataFrames, Tables

# Ruta de archivo y parámetros de filtro
file_path = "/home/joaquintschopp/buckets/b1/datasets/competencia_02_ct.csv.gz"
output_path = "/home/joaquintschopp/buckets/b1/datasets/competencia_julia_ct.csv"
training_months = [202108, 202107, 202106, 202105, 202104, 202103, 202102, 202101]

# Función para cargar, filtrar y escribir fragmentos
function process_and_filter(file_path, training_months, output_path)
    @info "Comienza carga y filtrado por fragmentos del dataset"

    # Abrir archivo para escritura con compresión
    CSV.open(output_path, write=true, compress=true) do writer
        first_chunk = true

        # Leer y procesar fragmentos del archivo de entrada
        for chunk in CSV.File(file_path; buffer=2^20, reusebuffer=true)
            df_chunk = DataFrame(chunk)

            # Filtrar fragmento
            filtered_chunk = df_chunk[df_chunk.foto_mes.∈training_months, :]

            # Escribir el fragmento filtrado en el archivo de salida
            CSV.write(writer, filtered_chunk, append=(!first_chunk))
            first_chunk = false
        end
    end

    @info "Filtrado y guardado completado en $output_path"
end

# Ejecutar el procesamiento
process_and_filter(file_path, training_months, output_path)


