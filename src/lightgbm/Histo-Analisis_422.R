# Cargar la librería data.table
library(data.table)
library(ggplot2)

# Leer el archivo
df <- fread("G:/Mi unidad/01-Maestria Ciencia de Datos/DMEyF/TPs/dmeyf-2024/exp/HT4220/HT4220.csv", 
            header = TRUE, sep = ";")

# Verificar los primeros registros
print(head(df))

# Mostrar los nombres de las columnas
print(colnames(df))

# Convertir la columna de ganancia a millones
df$ganancia_millones <- df$ganancia / 1e6

# Generar el gráfico
ggplot(df, aes(x = iteracion, y = ganancia_millones)) +
  geom_line() +                # Línea conectando los puntos
  geom_point() +               # Puntos de cada iteración
  labs(x = "Iteración", y = "Ganancia (millones)", title = "Curva de Ganancia por Iteración") +
  theme_minimal()