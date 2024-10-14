}
  cat("\n")

  # Ordenar predicciones por probabilidad
  ordenadas <- order(prediccion, decreasing = TRUE)

  # Crear un vector de 1 y 0
  predicted <- rep(0, length(prediccion))
  predicted[ordenadas[1:12500]] <- 1

  # Guardar predicciones en archivo CSV
  write.csv(data.frame(numero_de_cliente = dataset_test$numero_de_cliente, Predicted = predicted),
            file = paste0("predicciones_", sprintf("%03d", GLOBAL_iteracion), ".csv"), row.names = FALSE)

  # Esperar valor de ganancia por teclado
  ganancia_usuario <- as.numeric(readline(prompt = "Ingrese la ganancia: "))

  # voy grabando las mejores column importance
  if ( ganancia_usuario > GLOBAL_gananciamax) {
    GLOBAL_gananciamax <<- ganancia_usuario
    tb_importancia <- as.data.table(lgb.importance(modelo_train))

    fwrite(tb_importancia,
      file = paste0("impo_", sprintf("%03d", GLOBAL_iteracion), ".txt"),
      sep = "\t"
    )

    rm(tb_importancia)
  }