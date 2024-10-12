# Limpio la memoria
rm(list = ls()) # remove all objects
gc() # garbage collection

# Carga las librerías necesarias
require("data.table")
require("lightgbm")
require("ggplot2")
library(dplyr)
library(lubridate)

# Instalar las librerías necesarias
library(iml)

# Cargo el dataset donde voy a entrenar
dataset <- fread("G:/Mi unidad/01-Maestria Ciencia de Datos/DMEyF/TPs/dmeyf-2024/datasets/competencia_01_crudo.csv", stringsAsFactors = TRUE)
print(dim(dataset))


# Definir las columnas a eliminar
cols_a_eliminar <- c("cliente_vip", 
                      "internet", "cliente_edad", "cliente_antiguedad", 
                      "tcuentas", "tcallcenter", "ccallcenter_transacciones", 
                      "thomebanking", "chomebanking_transacciones", 
                      "Master_delinquency", "Master_status", "Master_Fvencimiento", 
                      "Master_Finiciomora", "Master_fultimo_cierre", "Master_fechaalta", 
                      "Visa_delinquency", "Visa_status", "Visa_Fvencimiento", 
                      "Visa_Finiciomora", "Visa_fultimo_cierre", "Visa_fechaalta")

# Eliminar las columnas especificadas
dataset <- dataset %>% select(-one_of(cols_a_eliminar))

# Eliminar columnas de tipo fecha
dataset <- dataset %>% select(-where(~ is.Date(.)))

# Comprobar el resultado
print(dim(dataset))

##########################################################################

#CORRIGO LAS ENCONTRADAS


PARAM <- list()

PARAM$driftingcorreccion <- "deflacion"#"ninguno"
#"deflacion"      = drift_deflacion(campos_monetarios),
#"dolar_blue"     = drift_dolar_blue(campos_monetarios),
#"dolar_oficial"  = drift_dolar_oficial(campos_monetarios),
#"UVA"            = drift_UVA(campos_monetarios),
#"estandarizar" 

#------------------------------------------------------------------------------
# valores financieros
# meses que me interesan
vfoto_mes <- c(
  202101, 202102, 202103,
  202104, 202105, 202106
)

# los valores que siguen fueron calculados por alumnos
#  si no esta de acuerdo, cambielos por los suyos

# momento 1.0  31-dic-2020 a las 23:59
vIPC <- c(
  0.9680542110, 0.9344152616, 0.8882274350,
  0.8532444140, 0.8251880213, 0.8003763543
)

vdolar_blue <- c(
  157.900000, 149.380952, 143.615385,
  146.250000, 153.550000, 162.000000
)

vdolar_oficial <- c(
  91.474000,  93.997778,  96.635909,
  98.526000,  99.613158, 100.619048
)

vUVA <- c(
  0.9669867858358365, 0.9323750098728378, 0.8958202912590305,
  0.8631993702994263, 0.8253893405524657, 0.7928918905364516
)

#------------------------------------------------------------------------------

drift_UVA <- function(campos_monetarios) {
  cat( "inicio drift_UVA()\n")
  
  dataset[tb_indices,
          on = c("foto_mes"),
          (campos_monetarios) := .SD * i.UVA,
          .SDcols = campos_monetarios
  ]
  
  cat( "fin drift_UVA()\n")
}
#------------------------------------------------------------------------------

drift_dolar_oficial <- function(campos_monetarios) {
  cat( "inicio drift_dolar_oficial()\n")
  
  dataset[tb_indices,
          on = c("foto_mes"),
          (campos_monetarios) := .SD / i.dolar_oficial,
          .SDcols = campos_monetarios
  ]
  
  cat( "fin drift_dolar_oficial()\n")
}
#------------------------------------------------------------------------------

drift_dolar_blue <- function(campos_monetarios) {
  cat( "inicio drift_dolar_blue()\n")
  
  dataset[tb_indices,
          on = c("foto_mes"),
          (campos_monetarios) := .SD / i.dolar_blue,
          .SDcols = campos_monetarios
  ]
  
  cat( "fin drift_dolar_blue()\n")
}
#------------------------------------------------------------------------------

drift_deflacion <- function(campos_monetarios) {
  cat( "inicio drift_deflacion()\n")
  
  dataset[tb_indices,
          on = c("foto_mes"),
          (campos_monetarios) := .SD * i.IPC,
          .SDcols = campos_monetarios
  ]
  
  cat( "fin drift_deflacion()\n")
}
#------------------------------------------------------------------------------

drift_estandarizar <- function(campos_drift) {
  
  cat( "inicio drift_estandarizar()\n")
  for (campo in campos_drift)
  {
    cat(campo, " ")
    dataset[, paste0(campo, "_normal") := 
              (get(campo) -mean(campo, na.rm=TRUE)) / sd(get(campo), na.rm=TRUE),
            by = "foto_mes"]
    
    dataset[, (campo) := NULL]
  }
  cat( "fin drift_estandarizar()\n")
}
#------------------------------------------------------------------------------



# tabla de indices financieros
tb_indices <- as.data.table( list( 
  "IPC" = vIPC,
  "dolar_blue" = vdolar_blue,
  "dolar_oficial" = vdolar_oficial,
  "UVA" = vUVA
  )
)

tb_indices$foto_mes <- vfoto_mes

tb_indices

##CAMPOS ENCONTRADOS CON DRIFTING
campos_monetarios <- as.character(c("mpayroll", "Visa_mlimitecompra", "Master_mfinanciacion_limite", 
                                    "mcomisiones_mantenimiento", "Visa_mfinanciacion_limite", 
                                    "Master_mlimitecompra", "Visa_msaldodolares", "Visa_mconsumosdolares", 
                                    "mcaja_ahorro_dolares", "mtransferencias_recibidas", 
                                    "mtarjeta_visa_consumo", "mpasivos_margen", "mcuentas_saldo", 
                                    "mextraccion_autoservicio", "mactivos_margen", "mrentabilidad_annual"))

# ordeno dataset
setorder(dataset, numero_de_cliente, foto_mes)

switch(PARAM$driftingcorreccion,
  "ninguno"        = cat("No hay correccion del data drifting"),
  "deflacion"      = drift_deflacion(campos_monetarios),
  "dolar_blue"     = drift_dolar_blue(campos_monetarios),
  "dolar_oficial"  = drift_dolar_oficial(campos_monetarios),
  "UVA"            = drift_UVA(campos_monetarios),
  "estandarizar"   = drift_estandarizar(campos_monetarios)
)

######################################

#EVALUACION DE DRIFT

######################################

# Filtrar los meses de interés (202104 y 202106)
dataset <- dataset[foto_mes %in% c(202104, 202106)]

dataset <- dataset[, -c("numero_de_cliente", "cpayroll_trx")]

# Convertir foto_mes en 0 (para 202104) y 1 (para 202106)
dataset[, foto_mes := ifelse(foto_mes == 202104, 0, 1)]

# Eliminar columnas constantes o vacías
cols_to_remove <- sapply(dataset, function(x) length(unique(x)) == 1 | all(is.na(x)))
dataset <- dataset[, !cols_to_remove, with=FALSE]

# Convertir variables categóricas a numéricas
dataset[] <- lapply(dataset, function(x) {
  if (is.factor(x) || is.character(x)) {
    return(as.numeric(as.factor(x)))
  }
  return(x)
})


print(dim(dataset))

# Asegúrate de que tienes todas las columnas menos 'foto_mes'
df <- dataset[, !names(dataset) %in% "foto_mes", with = FALSE]   # Excluyes 'foto_mes'
df$mes <- dataset$foto_mes  # Añades 'mes' como la columna objetivo al final


# Creación de un conjunto de entrenamiento y prueba
set.seed(123)
train_index <- sample(nrow(df), 0.8 * nrow(df))
test_index <- setdiff(1:nrow(df), train_index)
train_df <- df[train_index, ]
test_df <- df[test_index, ]


# Creación de matrices LightGBM
# Verifica si la columna "mes" existe en el dataframe antes de eliminarla
if("mes" %in% names(train_df)) {
  train_df_intermedio <- train_df[, !names(train_df) %in% "mes", with = FALSE]
} else {
  train_df_intermedio <- train_df
}

if("mes" %in% names(test_df)) {
  test_df_intermedio <- test_df[, !names(test_df) %in% "mes", with = FALSE]
} else {
  test_df_intermedio <- test_df
}

# Convertir a matriz
train_data <- as.matrix(train_df_intermedio)
test_data <- as.matrix(test_df_intermedio)

# Comprobar las dimensiones de las matrices resultantes
print(dim(train_data))
print(dim(test_data))

train_label <- train_df$mes
test_label <- test_df$mes

# Creación de los datasets LightGBM
dtrain <- lgb.Dataset(data = train_data, label = train_label)
dtest <- lgb.Dataset(data = test_data, label = test_label)

# Definición de los parámetros del modelo (ajustar min_data_in_leaf y min_data_in_bin)
params <- list(
  objective = "binary",
  metric = "binary_logloss",
  boosting_type = "gbdt",
  num_leaves = 31,
  learning_rate = 0.05,
  min_data_in_leaf = 1,  # Reducido para más flexibilidad
  min_data_in_bin = 1     # Reducido para más flexibilidad
)

# Entrenamiento del modelo
model <- lgb.train(
  params = params,
  data = dtrain,
  nrounds = 100,
  valids = list(test = dtest),
  eval_freq = 10,
  early_stopping_rounds = 10
)

# Verificar si el modelo fue entrenado correctamente
if (length(model$best_iter) > 0) {
  # Obtención del feature importance
  importance <- lgb.importance(model, percentage = TRUE)
  
  # Seleccionar las primeras 20 características más importantes
  top_20_importance <- head(importance, n = 20)
  
  # Verificar si el 'importance' contiene datos
  print(importance)
  
  # Si hay datos en 'importance', visualiza el gráfico
  if (nrow(importance) > 0) {
    grafico <- ggplot(top_20_importance, aes(x = reorder(Feature, Gain), y = Gain)) +
      geom_bar(stat = "identity") +
      coord_flip() +  # Para que el gráfico esté en horizontal
      labs(title = "Importancia de las variables", x = "Características", y = "Importancia") +
      theme_minimal()
    print(grafico)
  } else {
    print("No hay datos de importancia de características para graficar.")
  }
} else {
  print("El modelo no fue entrenado correctamente")
}


# Convertir de nuevo a dataframe
train_data_df <- as.data.frame(train_data)
test_data_df <- as.data.frame(test_data)

# Añadir nombres de columnas (si es que no lo tienen)
colnames(train_data_df) <- colnames(train_df_intermedio)
colnames(test_data_df) <- colnames(test_df_intermedio)

# Definir las etiquetas del conjunto de entrenamiento
train_label <- as.factor(train_df$mes)
test_label <- as.factor(test_df$mes)

# Convertir de nuevo a matriz para la predicción
train_matrix <- as.matrix(train_data_df)

# Definir una función personalizada para predecir con LightGBM
predict_function <- function(model, newdata) {
  # Convertir los datos nuevos a matriz, ya que LightGBM requiere ese formato
  newdata_matrix <- as.matrix(newdata)
  predict(model, newdata_matrix)
}

# Crear el predictor de iml usando el modelo LightGBM entrenado
predictor <- Predictor$new(
  model = model, 
  data = train_data_df,  # Usar los datos de entrenamiento
  y = train_label,       # Etiquetas correspondientes
  predict.fun = predict_function  # Función de predicción personalizada
)

# Elegir una observación de interés para analizar los valores SHAP
x_interest <- train_data_df[1, , drop = FALSE]

# Crear el objeto de Shapley
shap <- Shapley$new(predictor, x.interest = x_interest)

# Asumiendo que ya tienes el objeto `Shapley` calculado
shap_values <- shap$results

# Ver el formato de los resultados
head(shap_values)

# 1. Crear un data.frame a partir de los valores de Shapley y las variables
shap_df <- data.frame(
  Feature = shap_values$feature,
  SHAP_Value = shap_values$phi
)

# 2. Ordenar por el valor absoluto de SHAP
shap_df <- shap_df[order(abs(shap_df$SHAP_Value), decreasing = TRUE), ]

# 3. Seleccionar las primeras 10 y las últimas 10 variables
top_10 <- shap_df[1:10, ]
bottom_10 <- shap_df[(nrow(shap_df) - 9):nrow(shap_df), ]

# 4. Combinar las primeras y las últimas 10 variables
selected_shap_df <- rbind(top_10, bottom_10)

# 5. Generar el gráfico para estas variables
library(ggplot2)

plot_obj <-  ggplot(selected_shap_df, aes(x = reorder(Feature, SHAP_Value), y = SHAP_Value)) +
  geom_bar(stat = "identity") +
  coord_flip() +
  labs(
    title = "Top 10 y Bottom 10 Variables por Valor Shapley",
    x = "Variables",
    y = "Valor Shapley"
  ) +
  theme_minimal()

#print(plot_obj) 
