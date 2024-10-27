using DuckDB

# Conectar a una base de datos temporal en memoria
db = DuckDB.DB()

# Crear una vista de la tabla directamente desde el CSV sin cargarlo todo en memoria
DuckDB.register_filesystem(db)
DuckDB.load_csv(db, "/home/joaquintschopp/buckets/b1/datasets/competencia_02_ct.csv.gz", "competencia_data")

# Ejecutar consultas SQL directamente sobre el archivo CSV
result = DuckDB.query(db, "SELECT * FROM competencia_data WHERE foto_mes IN (202108, 202107, 202106, 202105, 202104, 202103, 202102, 202101)")

# Trabajar con `result` como un DataFrame
using DataFrames
filtered_data = DataFrame(result)

# Guardar el resultado si es necesario
CSV.write("/home/joaquintschopp/buckets/b1/datasets/competencia_julia_ct1.csv", filtered_data)

# Cerrar la conexión
DuckDB.close(db)