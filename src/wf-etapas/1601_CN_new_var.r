#!/usr/bin/env Rscript
cat( "ETAPA  z1601_CN_canaritos_asesinos.r  INIT\n")

# Workflow  Canaritos Asesinos

# inputs
#  * gran dataset
#  * especificaciones del ejercito de canaritos
# output  
#   dataset :
#     misma cantidad de registros
#     los valores de los campos no se modifican
#     atributos que sobrevivieron al ataque de los canaritos asesinos


# limpio la memoria
rm(list = ls(all.names = TRUE)) # remove all objects
gc(full = TRUE, verbose= FALSE) # garbage collection

require("data.table", quietly=TRUE)
require("yaml", quietly=TRUE)
require("Rcpp", quietly=TRUE)

require("lightgbm", quietly=TRUE)


#cargo la libreria
# args <- c( "~/labo2024ba" )
args <- commandArgs(trailingOnly=TRUE)
source( paste0( args[1] , "/src/lib/action_lib.r" ) )

#------------------------------------------------------------------------------
VPOS_CORTE <- c()

fganancia_lgbm_meseta <- function(probs, datos) {
  vlabels <- get_field(datos, "label")
  vpesos <- get_field(datos, "weight")

  tbl <- as.data.table(list(
    "prob" = probs,
    "gan" = ifelse(vlabels == 1 & vpesos > 1, envg$PARAM$train$gan1, envg$PARAM$train$gan0)
  ))

  setorder(tbl, -prob)
  tbl[, posicion := .I]
  tbl[, gan_acum := cumsum(gan)]
  setorder(tbl, -gan_acum) # voy por la meseta

  gan <- mean(tbl[1:500, gan_acum]) # meseta de tamaño 500

  pos_meseta <- tbl[1:500, median(posicion)]
  VPOS_CORTE <<- c(VPOS_CORTE, pos_meseta)

  return(list(
    "name" = "ganancia",
    "value" = gan,
    "higher_better" = TRUE
  ))
}
#------------------------------------------------------------------------------
# Elimina del dataset las variables que estan por debajo
#  de la capa geologica de canaritos
# se llama varias veces, luego de agregar muchas variables nuevas,
#  para ir reduciendo la cantidad de variables
# y así hacer lugar a nuevas variables importantes

GVEZ <- 1

CanaritosAsesinos <- function(
  canaritos_ratio,
  canaritos_desvios,
  canaritos_semilla) {

  cat( "inicio CanaritosAsesinos()\n")
  gc(verbose= FALSE)
  dataset[, clase01 := 0L ]
  dataset[ get(envg$PARAM$dataset_metadata$clase) %in% envg$PARAM$train$clase01_valor1, 
      clase01 := 1L ]

  set.seed(canaritos_semilla, kind = "L'Ecuyer-CMRG")
  for (i in 1:(ncol(dataset) * canaritos_ratio)) {
    dataset[, paste0("canarito", i) := runif(nrow(dataset))]
  }

  campos_buenos <- setdiff(
    colnames(dataset),
    c( campitos, "clase01")
  )

  azar <- runif(nrow(dataset))

  dataset[, entrenamiento :=
    as.integer( get(envg$PARAM$dataset_metadata$periodo) %in% envg$PARAM$train$training &
      (clase01 == 1 | azar < envg$PARAM$train$undersampling))]

  dtrain <- lgb.Dataset(
    data = data.matrix(dataset[entrenamiento == TRUE, campos_buenos, with = FALSE]),
    label = dataset[entrenamiento == TRUE, clase01],
    weight = dataset[
      entrenamiento == TRUE,
      ifelse(get(envg$PARAM$dataset_metadata$clase) %in% envg$PARAM$train$positivos, 1.0000001, 1.0)
    ],
    free_raw_data = FALSE
  )

  dvalid <- lgb.Dataset(
    data = data.matrix(dataset[get(envg$PARAM$dataset_metadata$periodo) %in% envg$PARAM$train$validation, campos_buenos, with = FALSE]),
    label = dataset[get(envg$PARAM$dataset_metadata$periodo) %in% envg$PARAM$train$validation, clase01],
    weight = dataset[
      get(envg$PARAM$dataset_metadata$periodo) %in% envg$PARAM$train$validation,
      ifelse( get(envg$PARAM$dataset_metadata$clase) %in% envg$PARAM$train$positivos, 1.0000001, 1.0)
    ],
    free_raw_data = FALSE
  )


  param <- list(
    objective = "binary",
    metric = "custom",
    first_metric_only = TRUE,
    boost_from_average = TRUE,
    feature_pre_filter = FALSE,
    verbosity = -100,
    seed = canaritos_semilla,
    max_depth = -1, # -1 significa no limitar,  por ahora lo dejo fijo
    min_gain_to_split = 0.0, # por ahora, lo dejo fijo
    lambda_l1 = 0.0, # por ahora, lo dejo fijo
    lambda_l2 = 0.0, # por ahora, lo dejo fijo
    max_bin = 31, # por ahora, lo dejo fijo
    num_iterations = 9999, # un numero grande, lo limita early_stopping_rounds
    force_row_wise = TRUE, # para que los alumnos no se atemoricen con  warning
    learning_rate = 0.065,
    feature_fraction = 1.0, # lo seteo en 1
    min_data_in_leaf = 260,
    num_leaves = 60,
    early_stopping_rounds = 200,
    num_threads = 1
  )

  set.seed(canaritos_semilla, kind = "L'Ecuyer-CMRG")
  modelo <- lgb.train(
    data = dtrain,
    valids = list(valid = dvalid),
    eval = fganancia_lgbm_meseta,
    param = param,
    verbose = -100
  )

  tb_importancia <- lgb.importance(model = modelo)
  tb_importancia[, pos := .I]
  
  #####################################################################################
  #New Variables
  #####################################################################################
  #aca tomo las 20 mas importantes.
  top_vars <- head(tb_importancia[order(-tb_importancia$Gain)], 20)
  
  # Recorremos top_vars y generamos nuevas columnas con las transformaciones
  for (var in top_vars$Feature) {
  # Logaritmo natural (agregar 1 para evitar log(0))
    dataset[[paste0(var, "_log")]] <- log(dataset[[var]] + 1)
    dataset[[paste0(var, "_log")]][is.infinite(dataset[[paste0(var, "_log")]]) | is.nan(dataset[[paste0(var, "_log")]])] <- 0
  
  # Raíz cuadrada
    dataset[[paste0(var, "_sqrt")]] <- sqrt(pmax(dataset[[var]], 0))  # Evitar valores negativos
    dataset[[paste0(var, "_sqrt")]][is.nan(dataset[[paste0(var, "_sqrt")]])] <- 0
  
  # Potencia al cuadrado
    dataset[[paste0(var, "_squared")]] <- dataset[[var]]^2
    dataset[[paste0(var, "_squared")]][is.nan(dataset[[paste0(var, "_squared")]])] <- 0
  
  # Ratio entre la variable y otra importante (ejemplo: dividimos entre la primera variable de top_vars)
    if (var != top_vars$Feature[1]) {
      dataset[[paste0(var, "_ratio_", top_vars$Feature[1])]] <- dataset[[var]] / (dataset[[top_vars$Feature[1]]] + 1e-6)
      dataset[[paste0(var, "_ratio_", top_vars$Feature[1])]][is.nan(dataset[[paste0(var, "_ratio_", top_vars$Feature[1])]]) | is.infinite(dataset[[paste0(var, "_ratio_", top_vars$Feature[1])]])] <- 0
    }
  
  # Diferencia absoluta respecto a la primera variable de top_vars
    if (var != top_vars$Feature[1]) {
      dataset[[paste0(var, "_diff_", top_vars$Feature[1])]] <- abs(dataset[[var]] - dataset[[top_vars$Feature[1]]])
      dataset[[paste0(var, "_diff_", top_vars$Feature[1])]][is.nan(dataset[[paste0(var, "_diff_", top_vars$Feature[1])]])] <- 0
    }
  }
  cat( "COMO VAS AHSTA ACA()\n")
# Recorremos top_vars y generamos la suma y promedio para cada cliente identificado por "numero_de_cliente" usando data.table

# Agrupamos por numero_de_cliente y calculamos la suma para cada variable en top_vars
  aggregated_sum <- dataset[, lapply(.SD, sum, na.rm = TRUE), by = numero_de_cliente, .SDcols = top_vars$Feature]
  setnames(aggregated_sum, old = names(aggregated_sum)[-1], new = paste0(top_vars$Feature, "_sum"))

# Agrupamos por numero_de_cliente y calculamos el promedio para cada variable en top_vars
  aggregated_mean <- dataset[, lapply(.SD, mean, na.rm = TRUE), by = numero_de_cliente, .SDcols = top_vars$Feature]
  setnames(aggregated_mean, old = names(aggregated_mean)[-1], new = paste0(top_vars$Feature, "_mean"))

# Unimos los resultados agregados al dataset original usando merge para evitar perder filas
  setkey(dataset, numero_de_cliente)
  setkey(aggregated_sum, numero_de_cliente)
  setkey(aggregated_mean, numero_de_cliente)

# Unimos sumas y promedios al dataset asegurando que no se pierdan registros
  dataset <- merge(dataset, aggregated_sum, by = "numero_de_cliente", all.x = TRUE)
  dataset <- merge(dataset, aggregated_mean, by = "numero_de_cliente", all.x = TRUE)

# Reemplazamos los posibles NA resultantes de las fusiones con 0
  # Reemplazamos los posibles NA resultantes de las fusiones con 0, solo en las columnas correspondientes a top_vars
  top_vars_cols <- c(paste0(top_vars$Feature, "_sum"), paste0(top_vars$Feature, "_mean"))
  dataset[, (top_vars_cols) := lapply(.SD, function(x) ifelse(is.na(x), 0, x)), .SDcols = top_vars_cols]

  cat( "y AHORA, COMO VAS AHSTA ACA()\n")

  # Generamos las sumas entre las variables originales de top_vars, cada una sumada contra las demás
  for (i in seq_along(top_vars$Feature)) {
    for (j in (i+1):length(top_vars$Feature)) {
      var1 <- top_vars$Feature[i]
      var2 <- top_vars$Feature[j]
      new_col_name <- paste0(var1, "_plus_", var2)
    
      # Verificamos que ambas columnas tengan datos antes de realizar la suma
      if (all(!is.na(dataset[[var1]])) && all(!is.na(dataset[[var2]]))) {
        dataset[[new_col_name]] <- dataset[[var1]] + dataset[[var2]]
        dataset[[new_col_name]][is.nan(dataset[[new_col_name]])] <- 0
      } else {
        dataset[[new_col_name]] <- 0  # Si alguna de las columnas tiene NA, asignamos 0
      }
    }
  }


  ## Fin Nuevas Variable  
  ###########################################################################################################
  fwrite(tb_importancia,
    file = paste0("impo_", GVEZ, ".txt"),
    sep = "\t"
  )

  GVEZ <<- GVEZ + 1

  umbral <- tb_importancia[
    Feature %like% "canarito",
    median(pos) + canaritos_desvios * sd(pos)
  ] # Atencion corto en la mediana mas desvios!!

  col_utiles <- tb_importancia[
    pos < umbral & !(Feature %like% "canarito"),
    Feature
  ]

  col_utiles <- unique(c(
    col_utiles,
    c(campitos, "mes")
  ))

  col_inutiles <- setdiff(colnames(dataset), col_utiles)

  #No elimino Nada
  #dataset[, (col_inutiles) := NULL]

  cat( "CREO FUE UN EXITO LA CREACION DE NUEVAS VARIABLES()\n")
}
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# Aqui empieza el programa
cat( "ETAPA  z1601_CN_canaritos_asesinos.r  START\n")
action_inicializar() 


envg$PARAM$CanaritosAsesinos$semilla <- envg$PARAM$semilla
  
# cargo el dataset donde voy a entrenar
# esta en la carpeta del exp_input y siempre se llama  dataset.csv.gz
# cargo el dataset
envg$PARAM$dataset <- paste0( "./", envg$PARAM$input, "/dataset.csv.gz" )
envg$PARAM$dataset_metadata <- read_yaml( paste0( "./", envg$PARAM$input, "/dataset_metadata.yml" ) )

cat( "lectura del dataset\n")
action_verificar_archivo( envg$PARAM$dataset )
cat( "Iniciando lectura del dataset\n" )
dataset <- fread(envg$PARAM$dataset)
cat( "Finalizada lectura del dataset\n" )


colnames(dataset)[which(!(sapply(dataset, typeof) %in% c("integer", "double")))]


GrabarOutput()

#--------------------------------------
# estas son las columnas a las que se puede agregar
#  lags o media moviles ( todas menos las obvias )

campitos <- c( envg$PARAM$dataset_metadata$primarykey,
  envg$PARAM$dataset_metadata$entity_id,
  envg$PARAM$dataset_metadata$periodo,
  envg$PARAM$dataset_metadata$clase )

campitos <- unique( campitos )

cols_lagueables <- copy(setdiff(
  colnames(dataset),
  envg$PARAM$dataset_metadata
))

# ordeno el dataset por primary key
#  es MUY  importante esta linea
# ordeno dataset
cat( "ordenado del dataset\n")
setorderv(dataset, envg$PARAM$dataset_metadata$primarykey)

#--------------------------------------------------------------------------
# Elimino las variables que no son tan importantes en el dataset
# with great power comes grest responsability

envg$OUTPUT$CanaritosAsesinos$ncol_antes <- ncol(dataset)
CanaritosAsesinos(
  canaritos_ratio = envg$PARAM$CanaritosAsesinos$ratio,
  canaritos_desvios = envg$PARAM$CanaritosAsesinos$desvios,
  canaritos_semilla = envg$PARAM$CanaritosAsesinos$semilla
)

envg$OUTPUT$CanaritosAsesinos$ncol_despues <- ncol(dataset)
GrabarOutput()

#------------------------------------------------------------------------------
# grabo el dataset
cat( "escritura del dataset\n")
cat( "Iniciando grabado del dataset\n" )
fwrite(dataset,
  file = "dataset.csv.gz",
  logical01 = TRUE,
  sep = ","
)
cat( "Finalizado grabado del dataset\n" )

# copia la metadata sin modificar
cat( "escritura de metadata\n")
write_yaml( envg$PARAM$dataset_metadata, 
  file="dataset_metadata.yml" )

#------------------------------------------------------------------------------

# guardo los campos que tiene el dataset
tb_campos <- as.data.table(list(
  "pos" = 1:ncol(dataset),
  "campo" = names(sapply(dataset, class)),
  "tipo" = sapply(dataset, class),
  "nulos" = sapply(dataset, function(x) {
    sum(is.na(x))
  }),
  "ceros" = sapply(dataset, function(x) {
    sum(x == 0, na.rm = TRUE)
  })
))

fwrite(tb_campos,
  file = "dataset.campos.txt",
  sep = "\t"
)

#------------------------------------------------------------------------------
cat( "Fin del programa\n")

envg$OUTPUT$dataset$ncol <- ncol(dataset)
envg$OUTPUT$dataset$nrow <- nrow(dataset)

envg$OUTPUT$time$end <- format(Sys.time(), "%Y%m%d %H%M%S")
GrabarOutput()

#------------------------------------------------------------------------------
# finalizo la corrida
#  archivos tiene a los files que debo verificar existen para no abortar

action_finalizar( archivos = c("dataset.csv.gz","dataset_metadata.yml")) 
cat( "ETAPA  z1601_CN_canaritos_asesinos.r  END\n")
