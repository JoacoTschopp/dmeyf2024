# Cargar las librerías necesarias
using CodecZlib
using DataFrames
using CSV
using Glob
using FilePaths
using LightGBM
using Distributed

# Listar y guardar los archivos *.model
println("Cargar los modelos de LGBM:")
modelos_path = "/home/joaquintschopp/buckets/b1/expw/FM-0005/"
model_files = glob("*.model", modelos_path)

# Crear un DataFrame para almacenar los nombres de archivos y las semillas
model_info = DataFrame(modelo=String[], semilla=String[])

# Extraer la semilla del nombre del modelo y guardar en el DataFrame
for model_file in model_files
    # Extraer la semilla, que se encuentra entre "s" y ".model" y tiene 6 dígitos
    match_result = match(r".*s(\d{6})\.model", model_file)
    if match_result !== nothing
        semilla = match_result.captures[1]
        push!(model_info, (model_file, semilla))
        println(semilla)
    end
end

# Cargar el dataset
println("Cargando dataset desde: /home/joaquintschopp/buckets/b1/datasets/competencia_02_ct.csv.gz")
dataset_path = "/home/joaquintschopp/buckets/b1/datasets/competencia_02_ct.csv.gz"
df_dataset = DataFrame(CSV.File(dataset_path))

# Crear un DataFrame para almacenar las predicciones
df_predictions = DataFrame(
    numero_de_cliente=df_dataset.numero_de_cliente,
    foto_mes=df_dataset.foto_mes,
    clase_ternaria=df_dataset.clase_ternaria
)

# Distribuir la carga de trabajo para predicciones con múltiples modelos
@everywhere function predict_for_model(df_dataset, model_file, semilla)
    booster = Booster(model_file)
    predictions = Float64[]
    for row in eachrow(df_dataset)
        input_data = DataFrame(row)[:, Not([:numero_de_cliente, :foto_mes, :clase_ternaria])]
        input_matrix = convert(Matrix{Float64}, input_data)
        push!(predictions, predict(booster, input_matrix)[1])
    end
    return predictions
end

# Crear un diccionario para almacenar predicciones por cada modelo
predictions_dict = Dict()

# Distribuir la predicción entre varios workers si es posible
for model_row in eachrow(model_info)
    model_file = model_row.modelo
    semilla = model_row.semilla
    predictions_dict["w1_s$semilla"] = @spawn predict_for_model(df_dataset, model_file, semilla)
end

# Recoger las predicciones y agregarlas al DataFrame
df_predictions = copy(df_predictions)
for (key, handle) in predictions_dict
    df_predictions[!, Symbol(key)] .= fetch(handle)
end

# Guardar el DataFrame resultante en un archivo de texto
output_path = "/home/joaquintschopp/buckets/b1/stacking/estructura_de_nivel1_w1.csv.gz"
CSV.write(output_path, df_predictions, delim=",")

# Mostrar mensaje de finalización
println("Predicciones guardadas en: ", output_path)