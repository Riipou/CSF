using Random
include("../algorithms/functions.jl")
include("../algorithms/GD.jl")

Random.seed!(2024)

U = randn(10,2)
V = randn(2,10)
M = (U*V).^2
U, V = init_matrix(M, 2, "random")
U,V = BLS(100000,M,U,V, alpha = 0.9999)
println(norm(M-(U*V).^2)/norm(M))