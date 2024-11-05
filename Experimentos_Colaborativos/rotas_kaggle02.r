# Este script carga el dataset competencia_02_ct.csv.gz, calcula la media de cada atributo por cada mes (foto_mes), 
# y genera gráficos de tendencia para cada atributo. 
# Los gráficos se guardan en un archivo PDF.

library(data.table)
library(ggplot2)
library(gridExtra)

file_path <- "~/buckets/b1/datasets/competencia_02_ct.csv.gz"

cat("Cargando el dataset, esto puede llevar algo de tiempo...\n")
dataset <- fread(file_path)

dataset <- dataset[, .SD, .SDcols = c('foto_mes', names(dataset)[sapply(dataset, is.numeric)])]

cat("Calculando la media por cada atributo y cada foto_mes...\n")
medias_por_mes <- dataset[, lapply(.SD, mean, na.rm = TRUE), by = foto_mes]

medias_por_mes_transpuesta <- melt(medias_por_mes, id.vars = "foto_mes")
resultado_df <- dcast(medias_por_mes_transpuesta, variable ~ foto_mes, value.var = "value")

print(resultado_df)

cat("Generando gráficos de tendencia para cada atributo...\n")
output_pdf <- "~/buckets/b1/exp/tendencia_Kaggle02.pdf"
pdf(output_pdf)

for (i in 1:nrow(resultado_df)) {
  atributo <- resultado_df$variable[i]
  valores <- as.numeric(resultado_df[i, -1])
  foto_mes <- colnames(resultado_df)[-1]
  
  df_grafico <- data.frame(foto_mes = as.factor(foto_mes), valor = valores)
  p <- ggplot(df_grafico, aes(x = foto_mes, y = valor, group = 1)) +
    geom_line(color = "blue") +
    geom_point() +
    ggtitle(paste("Tendencia del atributo:", atributo)) +
    xlab("Mes") +
    ylab("Media del atributo") +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
    scale_x_discrete(breaks = levels(df_grafico$foto_mes)[seq(1, length(levels(df_grafico$foto_mes)), by = 6)])
  
  print(p)
}

dev.off()

cat("Gráficos de tendencia guardados en:", output_pdf, "\n")
