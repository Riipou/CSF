include("../algorithms/function_CD.jl")
include("../algorithms/giant/GIANT.jl")

using .GIANT
using MAT
using Random
using SparseArrays
using Plots
using Statistics

function open_dataset(
    dataset::String)

    # Path of data set
    file_path = "data sets/$(dataset).mat"
    # Load the .mat file
    data = matread(file_path)

    if dataset == "CBCLfacialfeatures"
        # Access variables in the .mat file
        M = data["U"]
    elseif dataset == "CBCL"
        # Access variables in the .mat file
        M = data["X"]
    elseif dataset == "TDT2" || dataset == "TDT2compressed"
        # Access variables in the .mat file
        M = data["Xkl"]
        M = Matrix(M)
    end

    return M
end
function info_dataset(dataset)
    M = open_dataset(dataset)
    println(dataset)
    println(size(M))
    non_zero_elements = count(!iszero, M)
    println(non_zero_elements)
    density = non_zero_elements / length(M)
    println(density*100)
end
function CSFandNMF_test(
    max_time::Int,
    r::Int,
    nb_tests::Int,
    dataset::String;
    max_iterations::Int = 10000,
    submatrix::Bool = false)
    
    # Choice of random seed
    Random.seed!(2024)

    if dataset == "sparse"
        # Generate a sparse matrix of size 200-by-200 with density 0.01
        M = Matrix(sprandn(200, 200, 0.01))
    else
        # Open data set
        M = open_dataset(dataset)
    end

    # Usage of a submatrix
    if submatrix
        M = M[1:50,1:100]
    end

    open("results/sparse_dataset/$(dataset)_r=$(r)_submatrix=$(submatrix)_max_time=$(max_time).txt", "w") do file

        # Definition of variables
        errors = []
        errors_nmf = []

        m,n = size(M)
        U = zeros(m,r)
        V = zeros(r,n)
        best_U, worst_U = copy(U), copy(U)
        best_V, worst_V = copy(V), copy(V)
    
        # CSF loop
        for i in 1:nb_tests
            println(i)
            U, V = init_matrix(M, r, "random")
            U, V = coordinate_descent(max_iterations, M, U, V, max_time = max_time)
            if i == 1
                worst_U = U
                worst_V = V
            end
            push!(errors, norm(M - (U * V).^2)/norm(M))
            if norm(M - (U * V).^2)/norm(M) < norm(M - (best_U * best_V).^2)/norm(M)
                best_U = U
                best_V = V
            end
            if norm(M - (U * V).^2)/norm(M) > norm(M - (worst_U * worst_V).^2)/norm(M)
                worst_U = U
                worst_V = V
            end
        end

        # NMF loop
        for i in 1:nb_tests
            println(i)
            W, H = nmf(M, r, maxiter = 1000000)
            push!(errors_nmf, norm(M - W*H)/norm(M))
        end

        average_error = sum(errors)/nb_tests
        average_error_nmf = sum(errors_nmf)/nb_tests

        # SVD initialization
        U_0, V_0 = init_matrix(M, r, "SVD")
        # CSF
        U, V = coordinate_descent(max_iterations, M, U_0, V_0, max_time = max_time)
        # NMF
        W = copy(U_0)
        H = copy(V_0)
        # Best rank-one approximation
        W[:, 1] .= abs.(U_0[:, 1])
        H[1, :] .= abs.(V_0[1, :])
        # Next (r-1) rank-one factors
        i = 2
        j = 2
        while i <= r
            if i % 2 == 0
                W[:, i] .= max.(U_0[:, j], 0)
                H[i, :] .= max.(V_0[j, :], 0)
            else
                W[:, i] .= max.(-U_0[:, j], 0)
                H[i, :] .= max.(-V_0[j, :], 0)
                j += 1
            end
            i += 1
        end
        W, H = nmf(M, r, maxiter = 1000000, W0=W, H0=H)

        # Results
        write(file, "CSF (Random initialization)\n")
        write(file, "average error = $(average_error * 100)%\n")
        write(file, "standard deviation = $(std(errors)*100)%\n")
        write(file, "CSF (SVD initialization)\n")
        write(file, "error = $((norm(M - (U * V).^2)/norm(M)) * 100)%\n")
        write(file, "NMF (Random initialization)\n")
        write(file, "average error = $(average_error_nmf * 100)%\n")
        write(file, "standard deviation = $(std(errors_nmf)*100)%\n")
        write(file, "NMF (SVD initialization)\n")
        write(file, "error = $((norm(M - W*H)/norm(M))* 100)%")
        
        # Create a dictionary containing the matrices
        data_dict = Dict("bU" => best_U, "bV" => best_V, "wU" => worst_U, "wV" => worst_V)
        # Save the dictionary to a .mat file
        matwrite("results/sparse_dataset/matrices/$(dataset)_r=$(r)_submatrix=$(submatrix)_max_time=$(max_time).mat", data_dict)

        end
end

# datasets = ["CBCL", "CBCLfacialfeatures", "TDT2"]
# for dataset in datasets
#     info_dataset(dataset)
# end

# r_values = [10, 20, 49]
# dataset = "CBCL"
# for r in r_values
#     max_time = 60
#     nb_tests_value = 10
#     CSFandNMF_test(max_time, r, nb_tests_value, dataset)
# end

# r_values = [10, 20]
# dataset = "TDT2"
# for r in r_values
#     max_time = 60
#     nb_tests_value = 10
#     CSFandNMF_test(max_time, r, nb_tests_value, dataset)
# end

# r_values = [10, 20]
# dataset = "sparse"
# for r in r_values
#     max_time = 60
#     nb_tests_value = 10
#     CSFandNMF_test(max_time, r, nb_tests_value, dataset)
# end 

# r_values = [10, 20]
# dataset = "CBCLfacialfeatures"
# for r in r_values
#     max_time = 60
#     nb_tests_value = 10
#     CSFandNMF_test(max_time, r, nb_tests_value, dataset)
# end



# CSF = [69.02355281775884, 39.62659232108218, 24.74901295645741, 15.875810104379825]
# NMF = [89.6452978286602, 81.00895769209575, 72.72404199224063, 64.63757274336808]
# rank = [10, 20, 30, 40]

# plot(rank, CSF, label="CSF", xlabel="Rank", ylabel="Average Relative Error", linewidth=2, linestyle=:solid, marker=:circle, markersize=5)
# plot!(rank, NMF, label="NMF", linewidth=2, linestyle=:dash, marker=:square, markersize=5)
# xlabel!("Rank", titlefont=font(30))
# ylabel!("Average Relative Error", titlefont=font(30))
# plot!(legendfontsize=30)
# savefig("plot.png")
