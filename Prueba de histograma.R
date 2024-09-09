# Datos
x <- c(9000, 9500, 10000, 10500, 11000, 11500, 12000, 12500, 13000, 13500, 14000, 14500, 15000)
ya <- c(82038, 81758, 82178, 83367, 83437, 84417, 88711, 90997, 90181, 88734, 91791, 89084, 88337) #1?? Semilla
yb <- c(81408, 84347, 84184, 84254, 87544, 85701, 85561, 85771, 86937, 85281, 83997, 81548, 80708) #Bayesian Par??metros
yc <- c(80264, 80964, 81174, 79004, 79424, 85187, 88267, 89061, 90321, 90554, 91207, 88594, 88174) #2?? Semilla
yd <- c(79868, 86354, 86354, 85374, 86471, 88407, 89691, 89737, 87124, 84044, 80708, 77208, 78000) #2?? Semilla Bayesian

# Ajustar los valores de y a miles
ya_miles <- ya / 1000
yb_miles <- yb / 1000
yc_miles <- yc / 1000
yd_miles <- yd / 1000

# Crear el gr??fico de l??nea con eje y personalizado
plot(x, ya_miles, type = "o", col = "blue", lwd = 2,
     main = "Grafico de Ganancias",
     xlab = "Cortes Propuestos", ylab = "Valores Kaggle (en miles)",
     ylim = c(76, max(ya_miles, yb_miles, yc_miles, yd_miles)), # Ajustar el eje y
     pch = 16, cex = 1.2)

# A??adir la segunda, tercera y cuarta serie de datos
lines(x, yb_miles, type = "o", col = "red", lwd = 2, pch = 16)
lines(x, yc_miles, type = "o", col = "green", lwd = 2, pch = 16)
lines(x, yd_miles, type = "o", col = "purple", lwd = 2, pch = 16)

# A??adir una leyenda
legend("topleft", legend = c("1 Semilla Propuesta", "Bayesian Parametros", "2 Semilla Propuesta", "2 Semilla Bayesian"), 
       col = c("blue", "red", "green", "purple"), lty = 1, lwd = 2, pch = 16)

# Guardar el gr??fico en un archivo
png("grafico_lineas.png")
plot(x, ya_miles, type = "o", col = "blue", lwd = 2,
     main = "Grafico de Ganancias",
     xlab = "Cortes Propuestos", ylab = "Valores Kaggle (en miles)",
     ylim = c(76, max(ya_miles, yb_miles, yc_miles, yd_miles)), # Ajustar el eje y
     pch = 16, cex = 1.2)
lines(x, yb_miles, type = "o", col = "red", lwd = 2, pch = 16)
lines(x, yc_miles, type = "o", col = "green", lwd = 2, pch = 16)
lines(x, yd_miles, type = "o", col = "purple", lwd = 2, pch = 16)
legend("topright", legend = c("1 Semilla Propuesta", "Bayesian Parametros", "2 Semilla Propuesta", "2 Semilla Bayesian"), 
       col = c("blue", "red", "green", "purple"), lty = 1, lwd = 2, pch = 16)
dev.off()

