using CSV, DataFrames
using BenchmarkTools


periodo_anterior(x::Integer) = x % 100 > 1 ? x - 1 : 12 + (div(x, 100) - 1) * 100

###Apartado apra ejecucion en Goggle Cloud y Bukets
#df = CSV.read("gs://joaco333/datasets/competencia_01_crudo.csv", DataFrame)
df = CSV.read( "G:/Mi unidad/01-Maestria Ciencia de Datos/DMEyF/TPs/dmeyf-2024/datasets/competencia_01_crudo.csv", DataFrame )
#ruta = "/home/joaquintschopp/datasets/competencia_01_crudo.csv"

# Lee el archivo como un DataFrame
#df = CSV.read(ruta, DataFrame)

size(df)

sort!(df, [:numero_de_cliente, :foto_mes])

global periodo_ultimo = maximum(df.foto_mes)
global periodo_anteultimo = periodo_anterior(periodo_ultimo)

# assign most common class values
df.clase_ternaria = @. ifelse(df.foto_mes < periodo_anteultimo, "CONTINUA", missing)

# pre compute sequential time
periodo = @. div(df.foto_mes, 100) * 12 + df.foto_mes % 100

global last = nrow(df)

for i in 1:last
  if df.foto_mes[i] <= periodo_anteultimo && i < last &&
     (df.numero_de_cliente[i] != df.numero_de_cliente[i+1] || df.numero_de_cliente[i] == df.numero_de_cliente[i+1] && periodo[i+1] > periodo[i] + 1)
    df.clase_ternaria[i] = "BAJA+1"
  end

  if df.foto_mes[i] < periodo_anteultimo && i + 1 < last && df.numero_de_cliente[i] == df.numero_de_cliente[i+1] && periodo[i+1] == periodo[i] + 1 &&
     (df.numero_de_cliente[i+1] != df.numero_de_cliente[i+2] || df.numero_de_cliente[i+1] == df.numero_de_cliente[i+2] && periodo[i+2] > periodo[i+1] + 2)
    df.clase_ternaria[i] = "BAJA+2"
  end
end

CSV.write( "G:/Mi unidad/01-Maestria Ciencia de Datos/DMEyF/TPs/dmeyf-2024/datasets/competencia_01_julia.csv", df )
#CSV.write("/home/joaquintschopp/datasets/competencia_01_ct.csv", df)