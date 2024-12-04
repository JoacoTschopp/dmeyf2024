# Definir ruta del archivo log
log_file <- "~/buckets/b1/exp/lineademuerte/logs_lineademuerte.txt"

# Crear el archivo log y abrir la conexión
dir.create(dirname(log_file), recursive = TRUE, showWarnings = FALSE)  # Crear directorio si no existe
log_con <- file(log_file, open = "wt")

# Redirigir salida y errores a archivo log
sink(log_con, type = "output")
sink(log_con, type = "message")




# limpio la memoria
format(Sys.time(), "%a %b %d %X %Y")
rm(list = ls(all.names = TRUE)) # remove all objects
gc(full = TRUE, verbose= FALSE) # garbage collection

dir.create("~/buckets/b1/exp/lineademuerte/", showWarnings = FALSE)
setwd("~/buckets/b1/exp/lineademuerte/")

require("data.table")

# leo el dataset
dataset <- fread("~/buckets/b1/datasets/competencia_03_crudo.csv.gz")

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
  dataset[, paste0(vcol, "_delta2") := get(vcol) - get(paste0(vcol, "_lag2"))]
}

GLOBAL_semilla <- 214363

campos_buenos <- copy( setdiff(
    colnames(dataset), c("clase_ternaria"))
)

set.seed(GLOBAL_semilla, kind = "L'Ecuyer-CMRG")
dataset[, azar:=runif(nrow(dataset))]

dfuture <- dataset[foto_mes==202109]

# undersampling de los CONTINIA al 8%
dataset[, fold_train :=  foto_mes<= 202107 &
    (clase_ternaria %in% c("BAJA+1", "BAJA+2") |
     azar < 0.02 ) ]

dataset[, clase01 := ifelse( clase_ternaria=="CONTINUA", 0, 1 )]

require("lightgbm")

# dejo los datos en el formato que necesita LightGBM
dvalidate <- lgb.Dataset(
  data = data.matrix(dataset[foto_mes==202107, campos_buenos, with = FALSE]),
  label = dataset[foto_mes==202107, clase01],
  free_raw_data = TRUE
)

# aqui se hace la magia informatica con los pesos para poder reutilizar
#  el mismo dataset para training y final_train
dtrain <- lgb.Dataset(
  data = data.matrix(dataset[fold_train == TRUE, campos_buenos, with = FALSE]),
  label = dataset[fold_train == TRUE, clase01],
  weight = dataset[fold_train == TRUE, ifelse( foto_mes<=202106, 1.0, 0.0)],
  free_raw_data = TRUE
)

rm( dataset )
gc(full = TRUE, verbose= FALSE) # garbage collection

nrow( dfuture )
nrow( dvalidate )
nrow( dtrain )

# parametros basicos del LightGBM
param_basicos <- list(
    objective = "binary",
    metric = "auc",
    first_metric_only = TRUE,
    boost_from_average = TRUE,
    feature_pre_filter = FALSE,
    verbosity = -100,
    force_row_wise = TRUE, # para evitar warning
    seed = GLOBAL_semilla,
    max_bin = 31,
    learning_rate = 0.03,
    feature_fraction = 0.5
)


EstimarGanancia_AUC_lightgbm <- function(x) {

    message(format(Sys.time(), "%a %b %d %X %Y"))
    param_train <- list(
      num_iterations = 2048, # valor grande, lo limita early_stopping_rounds
      early_stopping_rounds = 200
    )

    param_completo <- c(param_basicos, param_train, x)

    modelo_train <- lgb.train(
      data = dtrain,
      valids = list(valid = dvalidate),
      eval = "auc", 
      param = param_completo,
      verbose = -100
    )

    AUC <- modelo_train$record_evals$valid$auc$eval[[modelo_train$best_iter]]

    # esta es la forma de devolver un parametro extra
    attr(AUC, "extras") <- list("num_iterations"= modelo_train$best_iter)
    
    rm(modelo_train)
    gc(full= TRUE, verbose= FALSE)
    
    return(AUC)
}





# configuración de la optimización bayesiana
require("DiceKriging")
require("mlrMBO")

configureMlr(show.learner.output = FALSE)

# configuro la busqueda bayesiana,  los hiperparametros que se van a optimizar
# por favor, no desesperarse por lo complejo
# función objetivo
obj.fun <- makeSingleObjectiveFunction(
    fn = EstimarGanancia_AUC_lightgbm, # la funcion que voy a maximizar
    minimize = FALSE, # estoy Maximizando AUC
    noisy = FALSE,
    par.set = makeParamSet(
       makeIntegerParam("num_leaves", lower = 8L, upper = 1024L),
       makeIntegerParam("min_data_in_leaf", lower = 64L, upper = 8192L)
    ),
    has.simple.signature = FALSE # paso los parametros en una lista
)



mejores_hiperparametros <- data.table(
    num_leaves = 966,
    min_data_in_leaf = 64,
    num_iterations = 2045
)
print(mejores_hiperparametros)

# configuración del modelo final
set_field(dtrain, "weight", rep(1.0, nrow(dtrain)))
param_final <- c(param_basicos, mejores_hiperparametros)




# Generación de modelos y predicciones
# Instalación y carga del paquete gmp si no está disponible
if (!require("gmp")) {
  install.packages("gmp", repos = "http://cran.us.r-project.org")
  library(gmp)
}
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


# Al final del script, cerrar las conexiones
sink(type = "output")
sink(type = "message")
close(log_con)