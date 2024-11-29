using Random
using Statistics
using DataFrames

# Definir una función auxiliar para los tiros
function ftirar(prob=0.4, qty=100)
  println(sum(rand() < prob for i in 1:qty))
end


for i in 1:100 ftirar() end