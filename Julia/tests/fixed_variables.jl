include("../algorithms/CD.jl")
include("../algorithms/functions.jl")
include("../algorithms/slackgon.jl")

using MAT
using Random

function fixed_slackngon(U, V,indices_U, indices_V)
    m, _ = size(U)
    M, _ = slackngon(m)
    U,V = coordinate_descent(1000000, M, U, V)
    println(norm(M - (U * V).^2) / norm(M))
    # Create a dictionary containing the matrices
    data_dict = Dict("U" => U, "V" => V)
    # Save the dictionary to a .mat file
    matwrite("results/fixed_variables/slackngon_$(m)x$(m)_r=$(size(U)[2])_error=$(norm(M - (U * V).^2) / norm(M)).mat", data_dict)
end

function fixed_slackgon_matrix(
    n::Int,
    r::Int)
    
    # Choice of random seed
    Random.seed!(2024)
    nb_tests = 10^4
    alpha = 0.99999
    M, _ = slackngon(n)
    open("results/fixed_variables/slackgon_matrix_$(n)x$(n)_aplha=$(alpha).txt", "w") do file
        nb_good = 0
        best_error = Inf
        best_U = zeros(n,r)
        best_V = zeros(r,n)
        for _ in 1:nb_tests
            U = randn(size(M, 1), r)
            V = zeros(r, size(M,2))
            for i in 1:r
                V[i,i] = 1
            end
            V[:,r+1:size(M,2)] = randn(r, size(M,2)-r)
            A = (U*V).^2
            lambda = sum(A[:].*M[:])/sum(A[:].^2)
            if lambda < 0
                U = rand(size(M, 1), r)
                V[:,r+1:size(M,2)] = randn(r, size(M,2)-r)
                A = (U*V).^2
                lambda = sum(A[:].*M[:])/sum(A[:].^2)
            end
            U *= lambda^(1/4)
            V *= lambda^(1/4)
            for i in 1:r
                V[i,i] = 1
            end
            indices_V = []
            for i in 1:r
                for j in 1:r
                    push!(indices_V, (i,j))
                end
            end
            indices_V = Vector{Tuple{Int64, Int64}}(indices_V)
            U, V = coordinate_descent_extrapoled(100000, M, U, V, alpha = alpha, indices_V = indices_V)
            error = norm(M - (U * V).^2) / norm(M)
            if error < norm(M - (best_U * best_V).^2) / norm(M)
                best_U = copy(U)
                best_V = copy(V)
            end
            if error < 1e-3
                nb_good += 1
            end
            if error < best_error
                best_error = error
            end
        end
        accuracy = nb_good / nb_tests
        write(file, "accuracy for r = $r : $(accuracy*100)%\n")
        write(file, "best error for r = $r : $best_error\n")
        # Create a dictionary containing the matrices
        data_dict = Dict("bU" => best_U, "bV" => best_V)
        # Save the dictionary to a .mat file
        matwrite("results/fixed_variables/$(n)x$(n)_r=$(r)_best_error=$(best_error).mat", data_dict)
    end
end

n = 8
r = 6

M, _ = slackngon(n)
file_name = "slackngon_8x8_r=6_error=4.175785360688005e-8"
# Path of initial U and V
file_path = "results/fixed_variables/$(file_name).mat"
# Load the .mat file
data = matread(file_path)
U = data["U"]
V = data["V"]
println("Matrix U:")
# Print each row of U row by row
for i in 1:size(U)[1]
    println(U[i, :])
end
println("Matrix V:")
# Print each row of U row by row
for i in 1:size(V)[1]
    println(V[i, :])
end

Xt = sign.(U*V).*sqrt(M)
println(rank(Xt))
indices_U = [(2,4)]
indices_V = [(5,2)]
for _ in 1:100
    fixed_slackngon(U,V,indices_U, indices_V)
end
#= fixed_slackgon_matrix(n,r) =#