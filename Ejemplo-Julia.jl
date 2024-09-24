ENV["LIGHTGBM_USE_SYSTEM_LIBS"] = "false"
using LightGBM



println(LightGBM.version())
# Parámetros del modelo
params = Dict("objective" => "binary", "learning_rate" => 0.1)


# Datos
X_train = rand(100, 10)  # 100 observaciones, 10 características
y_train = rand(0:1, 100) # Etiquetas binarias

# Crear el modelo
booster = LightGBM.Booster(params)

# Entrenar el modelo
LightGBM.fit!(booster, X_train, y_train)