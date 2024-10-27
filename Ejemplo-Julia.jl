using LightGBM
using DataFrames
using Statistics

# Definir el modelo
modelo = LGBMClassification()

# Definir el dataset
X = rand(100, 5)
y = rand(100)

# Crear un DataFrame con columnas individuales para cada característica
df = DataFrame(
    X1 = X[:, 1],
    X2 = X[:, 2],
    X3 = X[:, 3],
    X4 = X[:, 4],
    X5 = X[:, 5],
    y = y
)

# Definir la función de escalado
function escalar(X)
    mean_X = mean(X, dims=1)
    std_X = std(X, dims=1)
    return (X .- mean_X) ./ std_X
end

# Definir la función de entrenamiento
function entrenar(modelo, X, y)
    X_esc = escalar(X)
    fit!(modelo, X_esc, y)
end

# Definir la función de predicción
function predecir(modelo, X)
    X_esc = escalar(X)
    return predict(modelo, X_esc)
end

# Entrenar el modelo
entrenar(modelo, Matrix(df[:, 1:5]), df[:, 6])

# Predecir con el modelo
predicciones = predecir(modelo, Matrix(df[:, 1:5]))