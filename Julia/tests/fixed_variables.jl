include("../algorithms/CD.jl")
include("../algorithms/functions.jl")
include("../algorithms/slackgon.jl")

using MAT

function fixed_slackngon(U, V, indices_V, indices_U)
    m, _ = size(U)
    M, _ = slackngon(m)
    U,V = coordinate_descent(10000, M, U, V, indices_U = indices_U, indices_V = indices_V)
    println(norm(M - (U * V).^2) / norm(M))
    # Create a dictionary containing the matrices
    data_dict = Dict("U" => U, "V" => V)
    # Save the dictionary to a .mat file
    matwrite("results/fixed_variables/slackngon_$(m)x$(m)_r=$(size(U)[2])_error=$(norm(M - (U * V).^2) / norm(M)).mat", data_dict)
end

file_name = "slackngon_6x6_r=4_error=1.8827562241842413e-6"
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
U[6,4] = 0
U[1,1] = 0.05
V[1,3] = 1.78
V[3,6] = 2
V[1,1] = 2
V[3,1] = 1.7
indices_U = [(1,1), (6,4)]
indices_V = [(1,1), (1,3), (3,1), (3,6)]
fixed_slackngon(U,V, indices_V, indices_U)