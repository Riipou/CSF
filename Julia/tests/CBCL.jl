using MAT
include("../algorithms/function_CD.jl")
#Alors, 361x2429 ce sera peut-être trop grand pour ton algo, donc tu peux sélectionner une sous-matrice de taille 50x100 par exemple.

function CBCL_test()

    file_path = "data sets/CBCL.mat"

    # Load the .mat file
    data = matread(file_path)

    # Access variables in the .mat file
    M = data["X"]

    open("results/accuracy_CBCL_test.txt", "w") do file
        r = 10
        nb_tests = 20
        max_iterations = 10000
        alpha = 0.9999
        nb_good = 0
        nb_good_nmf = 0
        best_error = Inf
        best_error_NMF = Inf
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
        accuracy = nb_good/nb_tests
        accuracy_nmf = nb_good_nmf/nb_tests
        write(file, "CSF\n")
        write(file, "accuracy=$(accuracy * 100)%\n")
        write(file, "best_error=$(best_error)\n")
        write(file, "NMF\n")
        write(file, "accuracy=$(accuracy_nmf * 100)%\n")
        write(file, "accuracy=$(best_error_NMF * 100)%\n")
    end
end
CBCL_test()