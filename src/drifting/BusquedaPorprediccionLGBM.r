# Carga de librerías
library(lightgbm)
library(shap)

# Carga de datos
dataset <- fread("./datasets/competencia_01.csv")

# Creación de un data.frame con los datos y el mes como variable objetivo
df <- data.frame(
  mes = dataset$foto_mes,
  dataset[, -which(names(dataset) %in% c("foto_mes"))]
)

# Creación de un conjunto de entrenamiento y prueba
set.seed(123)
train_index <- sample(nrow(df), 0.8*nrow(df))
test_index <- setdiff(1:nrow(df), train_index)
train_df <- df[train_index, ]
test_df <- df[test_index, ]

# Entrenamiento del modelo
params <- list(
  objective = "multiclass",
  num_class = 2,
  metric = "multi_logloss",
  boosting_type = "gbdt",
  num_leaves = 31,
  learning_rate = 0.05
)
model <- lgb.train(
  data = train_df,
  params = params,
  label = train_df$mes,
  categorical_feature = "mes",
  num_iterations = 100
)

# Predicción en el conjunto de prueba
predicciones <- predict(model, test_df, type = "response")

# Análisis de la importancia de las variables utilizando SHAP values
shap_values <- shap(model, test_df)
shap_values_df <- data.frame(
  variable = rownames(shap_values),
  shap_value = shap_values
)

# Visualización de los SHAP values
ggplot(shap_values_df, aes(x = variable, y = shap_value)) +
  geom_bar(stat = "identity") +
  labs(title = "SHAP values", x = "Variable", y = "SHAP value") +
  theme_classic()