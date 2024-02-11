include("../algorithms/function_CD.jl")
include("../algorithms/giant/GIANT.jl")

using .GIANT
using MAT
using Random

function CBCL_test(
    alpha::Float64,
    r::Int,
    nb_tests::Int;
    max_iterations::Int = 10000,
    submatrix::Bool = false)
    
    # Choice of random seed
    Random.seed!(2024)
    # Path of data set
    file_path = "data sets/CBCL.mat"

    # Load the .mat file
    data = matread(file_path)

    # Access variables in the .mat file
    M = data["X"]

    # Usage of a submatrix
    if submatrix
        M = M[1:50,1:100]
    end

    open("results/accuracy_CBCL_test_submatrix=$(submatrix)_alpha=$(alpha).txt", "w") do file

        # Definition of variables
        nb_good = 0
        nb_good_nmf = 0
        best_error = Inf
        best_error_nmf = Inf
        m,n = size(M)
        U = zeros(m,r)
        V = zeros(r,n)

        # CSF loop
        for i in 1:nb_tests
            println(i)
            U, V = init_matrix(M, r, "random")
            U, V= coordinate_descent(max_iterations, M, U, V, alpha)
            if norm(M - (U * V).^2) / norm(M) < 1e-3
                nb_good += 1
            end
            if norm(M - (U * V).^2)/norm(M) < best_error
                best_error = norm(M - (U * V).^2)/norm(M)
            end
        end
        best_U = U
        best_V = V
        
        # NMF loop
        for i in 1:nb_tests
            println(i)
            W, H = nmf(M,r)
            if norm(M - W*H)/ norm(M) < 1e-3
                nb_good_nmf += 1
            end
            if norm(M - W*H)/norm(M) < best_error_nmf
                best_error_nmf = norm(M - W*H)/norm(M)
            end
        end

        U, V = init_matrix(M, r, "SVD")
        U, V= coordinate_descent(max_iterations, M, U, V, alpha)

        # Calculation of accuracies
        accuracy = nb_good/nb_tests
        accuracy_nmf = nb_good_nmf/nb_tests

        # Results
        write(file, "CSF (Random initialization)\n")
        write(file, "accuracy=$(accuracy * 100)%\n")
        write(file, "best_error=$(best_error* 100)%\n")
        write(file, "CSF (SVD initialization)\n")
        write(file, "error svd : $((norm(M - (U * V).^2)/norm(M)) * 100)%\n")
        write(file, "NMF\n")
        write(file, "accuracy=$(accuracy_nmf * 100)%\n")
        write(file, "accuracy=$(best_error_nmf * 100)%\n")
        write(file,"U : $(best_U)\n")
        write(file,"V : $(best_V)\n")
    end
end
alpha_value = 0.9999
r_value = 10
nb_tests_value = 20
CBCL_test(alpha_value, r_value, nb_tests_value)