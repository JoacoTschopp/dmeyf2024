# limpio la memoria
format(Sys.time(), "%a %b %d %X %Y")
rm(list = ls(all.names = TRUE)) # remove all objects
gc(full = TRUE, verbose= FALSE) # garbage collection

dir.create("~/buckets/b1/exp/lineademuerte/", showWarnings = FALSE)
setwd("~/buckets/b1/exp/lineademuerte/")

require("data.table")

# leo el dataset
dataset <- fread("~/buckets/b1/datasets/competencia_03_ct.csv.gz")

# calculo el periodo0 consecutivo
setorder(dataset, numero_de_cliente, foto_mes)
dataset[, periodo0 := as.integer(foto_mes/100) * 12 + foto_mes %% 100]

# calculo topes
periodo_ultimo <- dataset[, max(periodo0)]
periodo_anteultimo <- periodo_ultimo - 1

# calculo los leads de orden 1 y 2
dataset[, c("periodo1", "periodo2") := shift(periodo0, n = 1:2, fill = NA, type = "lead"), by = numero_de_cliente]

# asignar valores de clase más comunes = "CONTINUA"
dataset[periodo0 < periodo_anteultimo, clase_ternaria := "CONTINUA"]

# calculo BAJA+1
dataset[periodo0 < periodo_ultimo & (is.na(periodo1) | periodo0 + 1 < periodo1), clase_ternaria := "BAJA+1"]

# calculo BAJA+2
dataset[periodo0 < periodo_anteultimo & (periodo0 + 1 == periodo1) & (is.na(periodo2) | periodo0 + 2 < periodo2), clase_ternaria := "BAJA+2"]

dataset[, c("periodo0", "periodo1", "periodo2") := NULL]

# resumen de datos
tbl <- dataset[, .N, list(foto_mes, clase_ternaria)]
setorder(tbl, foto_mes, clase_ternaria)
print(tbl)

# Feature Engineering Histórico
cols_lagueables <- copy(setdiff(colnames(dataset), c("numero_de_cliente", "foto_mes", "clase_ternaria")))

dataset[, paste0(cols_lagueables, "_lag1") := shift(.SD, 1, NA, "lag"), by = numero_de_cliente, .SDcols = cols_lagueables]

dataset[, paste0(cols_lagueables, "_lag2") := shift(.SD, 2, NA, "lag"), by = numero_de_cliente, .SDcols = cols_lagueables]

# agrego los delta lags de orden 1
for (vcol in cols_lagueables) {
  dataset[, paste0(vcol, "_delta1") := get(vcol) - get(paste0(vcol, "_lag1"))]
}

# configuración de la optimización bayesiana
library(mlrMBO)

# función objetivo
ojb.fun <- makeSingleObjectiveFunction(
  fn = function(x) {
    # lógica de evaluación del modelo
    # placeholder para función de evaluación
    return(runif(1))
  },
  par.set = makeParamSet(
    makeIntegerParam("num_leaves", lower = 8L, upper = 1024L),
    makeIntegerParam("min_data_in_leaf", lower = 64L, upper = 8192L)
  ),
  has.simple.signature = FALSE
)

# cada 600 segundos guardo el resultado intermedio
ctrl <- makeMBOControl(
  save.on.disk.at.time = 600,
  save.file.path = "lineademuerte.RDATA"
)

# indico la cantidad de iteraciones que va a tener la Optimización Bayesiana
ctrl <- setMBOControlTermination(ctrl, iters = 10)

# defino el método estándar para la creación de los puntos iniciales
ctrl <- setMBOControlInfill(ctrl, crit = makeMBOInfillCritEI())

# configuración del modelo surrogate
surr.km <- makeLearner("regr.km", predict.type = "se", covtype = "matern3_2", control = list(trace = TRUE))

# correr optimización bayesiana
bayesiana_salida <- mbo(ojb.fun, learner = surr.km, control = ctrl)

# obtener mejores hiperparámetros
tb_bayesiana <- as.data.table(bayesiana_salida$opt.path)
# Verificar si la columna 'num_iterations' existe antes de ordenar
if (!"num_iterations" %in% colnames(tb_bayesiana)) {
  tb_bayesiana[, num_iterations := NA]  # Agregar columna vacía si no existe
}
setorder(tb_bayesiana, -y)
mejores_hiperparametros <- tb_bayesiana[1, list(num_leaves, min_data_in_leaf, num_iterations)]
print(mejores_hiperparametros)

# configuración del modelo final
set_field(dtrain, "weight", rep(1.0, nrow(dtrain)))
param_final <- c(param_basicos, mejores_hiperparametros)

# Generación de modelos y predicciones
library(gmp)
set.seed(214363)
semillas <- as.numeric(sapply(1:100, function(x) nextprime(sample(100000:999999, 1))))

predicciones_list <- list()

for (semilla in semillas) {
  set.seed(semilla)
  final_model <- lgb.train(data = dtrain, param = param_final, verbose = -100)
  prediccion <- predict(final_model, data.matrix(dfuture[, campos_buenos, with = FALSE]))
  predicciones_list[[as.character(semilla)]] <- prediccion
}

# Guardar predicciones
predicciones_df <- as.data.table(predicciones_list)
colnames(predicciones_df) <- paste0("m_", semillas)

fwrite(predicciones_df, file = "~/buckets/b1/exp/lineademuerte/predicciones_100_modelos.csv")

# Separar la generación de la tabla de entrega
tb_entrega <- dfuture[, list(numero_de_cliente)]
tb_entrega[, prob := predicciones_list[[as.character(semillas[1])]]]
setorder(tb_entrega, -prob)
tb_entrega[, prob := NULL]
tb_entrega[, Predicted := 0L]
tb_entrega[1:11000, Predicted := 1L]

# guardar la entrega
fwrite(tb_entrega, file = "lineademuerte_11000.csv")

# marcar el fin del script
format(Sys.time(), "%a %b %d %X %Y")