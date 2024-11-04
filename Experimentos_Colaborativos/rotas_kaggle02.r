# Cargar las librerías necesarias
library(data.table)
library(ggplot2)
library(gridExtra)

# Especificar la ruta del archivo
file_path <- "~/bukets/b1/datasets/competencia_02_ct.csv.gz"

# Cargar el dataset en un data.table
cat("Cargando el dataset, esto puede llevar algo de tiempo...\n")
dataset <- fread(file_path)

# Listar la cantidad de atributos (columnas) 
cantidad_atributos <- ncol(dataset)
cat("Cantidad de atributos (columnas) en el dataset: ", cantidad_atributos, "\n")

# Contar la cantidad de clientes por cada "foto_mes"
cat("Contando la cantidad de clientes por cada foto_mes...\n")
cantidad_clientes_por_mes <- dataset[, .N, by = foto_mes]

# Mostrar la cantidad de clientes por mes
print(cantidad_clientes_por_mes)

# Calcular la media por cada atributo y cada foto_mes
cat("Calculando la media por cada atributo y cada foto_mes...\n")
medias_por_mes <- dataset[, lapply(.SD, mean, na.rm = TRUE), by = foto_mes, .SDcols = setdiff(names(dataset), "foto_mes")]

# Transponer el data.table para tener 'foto_mes' como columnas y atributos como filas
medias_por_mes_transpuesta <- melt(medias_por_mes, id.vars = "foto_mes")
resultado_df <- dcast(medias_por_mes_transpuesta, variable ~ foto_mes, value.var = "value")

# Mostrar el nuevo DataFrame con medias por cada foto_mes
print(resultado_df)

# Crear gráficos de tendencia por cada atributo y guardarlos en un PDF
cat("Generando gráficos de tendencia para cada atributo...\n")
output_pdf <- "~/buckets/b1/exp/tendencia_Kaggle02.pdf"
pdf(output_pdf)

for (i in 1:nrow(resultado_df)) {
  # Extraer los datos del atributo actual
  atributo <- resultado_df$variable[i]
  valores <- as.numeric(resultado_df[i, -1])
  foto_mes <- colnames(resultado_df)[-1]
  
  # Crear el gráfico de tendencia
  df_grafico <- data.frame(foto_mes = as.numeric(foto_mes), valor = valores)
  p <- ggplot(df_grafico, aes(x = foto_mes, y = valor)) +
    geom_line(color = "blue") +
    geom_point() +
    ggtitle(paste("Tendencia del atributo:", atributo)) +
    xlab("Mes") +
    ylab("Media del atributo") +
    theme_minimal()
  
  # Dibujar el gráfico en el PDF
  print(p)
}

# Cerrar el archivo PDF
dev.off()

cat("Gráficos de tendencia guardados en:", output_pdf, "\n")