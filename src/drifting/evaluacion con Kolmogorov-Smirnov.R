# Limpio la memoria
rm(list = ls()) # remove all objects
gc() # garbage collection

# Carga las librerías necesarias
# Cargar el dataset
library(data.table)
library(dplyr)

dataset <- fread("G:/Mi unidad/01-Maestria Ciencia de Datos/DMEyF/TPs/dmeyf-2024/datasets/competencia_01_crudo.csv", stringsAsFactors = TRUE)
print(dim(dataset))

# Separar los datos según los periodos (foto_mes)
df_202104 <- dataset %>% filter(foto_mes == 202104)
df_202106 <- dataset %>% filter(foto_mes == 202106)

# Remover columnas no numéricas o irrelevantes
df_202104 <- df_202104 %>% select_if(is.numeric)
df_202106 <- df_202106 %>% select_if(is.numeric)

# Función para aplicar el test de Kolmogorov-Smirnov entre dos distribuciones
ks_test_results <- sapply(names(df_202104), function(col) {
  ks_test <- ks.test(df_202104[[col]], df_202106[[col]])
  return(ks_test$p.value)
})

# Crear un DataFrame con los resultados
ks_results_df <- data.frame(
  Variable = names(ks_test_results),
  P_Value = ks_test_results
)

# Ordenar por el valor p para ver cuáles difieren más
ks_results_df <- ks_results_df %>%
  arrange(P_Value)

# Mostrar los resultados del KS test
print(ks_results_df)

# Identificar qué variables tienen un P-Value por debajo del nivel de significancia (por ejemplo 0.05)
drifted_variables <- ks_results_df %>% filter(P_Value < 0.01)
print("Variables con data drift detectado:")
print(drifted_variables)