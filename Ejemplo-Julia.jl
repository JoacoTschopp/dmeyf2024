using Pkg
Pkg.add("MLJ")
Pkg.add("LightGBM")
Pkg.add("DataFrames")
Pkg.add("Random")

using DataFrames
using Random
using MLJ
using LightGBM


# Generar un dataset de prueba con 1000 filas y 5 columnas
Random.seed!(123)
n_filas = 1000
n_columnas = 5
X = rand(n_filas, n_columnas)
y = rand(n_filas)

# Convertir a DataFrame
df = DataFrame(X, :auto)
rename!(df, [:X1, :X2, :X3, :X4, :X5])
df[!, :y] = y



# Definir el modelo LightGBM
modelo = @load LightGBMClassifier

# Definir el pipeline
pipeline = @pipeline(
    StandardScaler(),
    modelo
)

# Preparar los datos para el entrenamiento
X = select!(df, Not(:y))
y = df[!, :y]

# Entrenar el modelo
mach = machine(pipeline, X, y)
fit!(mach)

# Evaluar el modelo
y_pred = predict(mach, X)
accuracy = mean(y_pred .== y)
println("Precisión: ", accuracy)