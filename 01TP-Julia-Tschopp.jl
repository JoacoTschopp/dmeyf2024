using CategoricalArrays


using CSV, DataFrames, Random, Statistics
using Primes
using DecisionTree, Impute
using Base.Threads

using Distributed

df = CSV.read("G:/Mi unidad/01-Maestria Ciencia de Datos/DMEyF/TPs/dmeyf-2024/datasets/competencia_01_julia.csv", DataFrame)

names(df)

println(length(unique(df.numero_de_cliente)))

function encontrar_registros_faltantes(df)
    registros_faltantes = []
    grupos = groupby(df, :numero_de_cliente)

    for grupo in grupos
        registros_cliente = grupo.foto_mes
        registros_cliente = sort(registros_cliente)

        if length(registros_cliente) == 6 && all(diff(registros_cliente) .== 1)
            continue
        end
        if length(registros_cliente) == 5 && (registros_cliente[1] == registros_cliente[end] - 4 || registros_cliente[end] == registros_cliente[1] + 4)
            continue
        end
        if any(diff(registros_cliente) .> 1) 
            push!(registros_faltantes, (grupo.numero_de_cliente[1]))
        end
    end

    return registros_faltantes
end

registros_faltantes = encontrar_registros_faltantes(df)
println(registros_faltantes)

@benchmark encontrar_registros_faltantes(df)

println(length(registros_faltantes))
registros_faltantes = [Int64(x) for x in registros_faltantes]
println(registros_faltantes)

function EliminarVacios(df, registros_faltantes)
    print("cantidad de registros de Train: ")
    println(size(df))
    # Eliminar los registros de estos clientes del df_train
    df = df[.!in.(df.numero_de_cliente, Ref(registros_faltantes)), :]
    print("cantidad de registros de Train: ")
    println(size(df))

    return df
end

function AgregarPerdidos(df, registros_faltantes)
    print("cantidad de registros de Train: ")
    println(size(df))

    # Crear un DataFrame vacío para almacenar los registros faltantes
    df_faltantes = DataFrame()

    # Iterar sobre los registros faltantes
    for registro in registros_faltantes
        # Buscar el registro correspondiente en el DataFrame original
        registro_original = df[df.numero_de_cliente .== registro, :]

        # Agregar una copia del registro original con 'foto_mes' faltante
        registro_faltante = deepcopy(registro_original)
        registro_faltante.foto_mes .= missing

        # Agregar el registro faltante al DataFrame de registros faltantes
        df_faltantes = vcat(df_faltantes, registro_faltante)
    end

    # Agregar los registros faltantes al DataFrame original
    df = vcat(df, df_faltantes)

    print("cantidad de registros de Train: ")
    println(size(df))
    return df
end

# Dividir el DataFrame en entrenamiento y prueba
df_train = filter(row -> row.foto_mes <= 202104, df)
df_test = filter(row -> row.foto_mes == 202106, df)

#si eliminamos
df_train = EliminarVacios(df_train, registros_faltantes)

#si agregamos los registros faltnates de esos 23...
#df_train = AgregarPerdidos(df_train)


#entrenameitno y target
X_train = select(df_train, Not(:clase_ternaria))
y_train = df_train.clase_ternaria


####----------------Control Missing
# Crear un DataFrame para mostrar los resultados
resultadoX_train = DataFrame(columna = String[], valores_faltantes = Int[])


# Iterar sobre cada columna del DataFrame original
for col in names(X_train)
    # Contar los valores faltantes en la columna actual
    missing_count = sum(ismissing.(X_train[!, col]))
    
    # Añadir el resultado al DataFrame de resultados
    push!(resultadoX_train, (col, missing_count))
end

# Contar los valores faltantes en la columna actual
resultadoy_train = sum(ismissing.(y_train))

# Mostrar el resultado
println(resultadoX_train)
print("Valores faltantes en y_train: ")
println(resultadoy_train)

CSV.write( "G:/Mi unidad/01-Maestria Ciencia de Datos/DMEyF/competencia_01_julia01(preProc).csv", df )