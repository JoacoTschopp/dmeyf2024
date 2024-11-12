# Cargar las librerías necesarias
using DataFrames
using CSV
using Glob
using FilePaths
using LightGBM

# Listar y guardar los archivos *.model
modelos_path = "~/buckets/b1/expw/FM-0005/*.model"
model_files = glob("*.model", modelos_path)

# Crear un DataFrame para almacenar los nombres de archivos y las semillas
model_info = DataFrame(modelo=String[], semilla=String[])

# Extraer la semilla del nombre del modelo y guardar en el DataFrame
for model_file in model_files
    # Extraer la semilla, que se encuentra entre "s" y ".model" y tiene 6 dígitos
    match_result = match(r"s(\d{6})\.model", model_file)
    if match_result !== nothing
        semilla = match_result.match
        push!(model_info, (model_file, semilla))
    end
end

# Cargar el dataset
println("Cargando dataset desde: ~/buckets/b1/datasets/competencia_02_ct.csv.gz")
dataset_path = "~/buckets/b1/datasets/competencia_02_ct.csv.gz"
df_dataset = DataFrame(CSV.File(dataset_path))

# Crear un DataFrame para almacenar las predicciones
df_predictions = DataFrame(
    numero_de_cliente=df_dataset.numero_de_cliente,
    foto_mes=df_dataset.foto_mes,
    clase_ternaria=df_dataset.clase_ternaria
)

# Realizar predicciones con cada modelo para cada fila del dataset
for row in eachrow(df_dataset)
    for model_row in eachrow(model_info)
        model_file = model_row.modelo
        semilla = model_row.semilla

        # Cargar el modelo LightGBM
        booster = Booster(model_file)

        # Preparar la fila para la predicción
        input_data = DataFrame(row)[:, Not([:numero_de_cliente, :foto_mes, :clase_ternaria])]
        input_matrix = convert(Matrix{Float64}, input_data)

        # Realizar la predicción
        prediccion = predict(booster, input_matrix)

        # Agregar la predicción al DataFrame con un nombre específico
        df_predictions[!, Symbol("w1_s$semilla")] = prediccion[1]
    end
end

# Guardar el DataFrame resultante en un archivo de texto
output_path = "~/buckets/b1/expw/estructura_de_nivel1_w1.txt"
CSV.write(output_path, df_predictions, delim=",")

# Mostrar mensaje de finalización
println("Predicciones guardadas en: ", output_path)