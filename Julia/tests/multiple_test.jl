include("../algorithms/CD.jl")
include("../algorithms/functions.jl")

using Random

function squared_factorisation(
    m::Int,
    n::Int,
    r::Int,
    nb_tests::Int,
    alpha::Float64,
    extrapoled::Bool)

    max_iterations = 10000
    nb_good_rand = 0
    nb_good_svd = 0
    
    for _ in 1:nb_tests
        U = randn(m, r)
        V = randn(r, n)
        M = (U * V).^2
        U_rand, V_rand = init_matrix(M, r, "random")
        U_svd, V_svd = init_matrix(M, r, "SVD")
        if extrapoled
            U_rand, V_rand = coordinate_descent_extrapoled(max_iterations,M,U_rand,V_rand,alpha = alpha)
            U_svd, V_svd = coordinate_descent_extrapoled(max_iterations, M, U_svd, V_svd, alpha = alpha)
        else
            U_rand, V_rand = coordinate_descent(max_iterations,M,U_rand,V_rand,alpha = alpha)
            U_svd, V_svd = coordinate_descent(max_iterations, M, U_svd, V_svd, alpha = alpha)
        end

        if norm(M - (U_rand * V_rand).^2) / norm(M) < 1e-3
            nb_good_rand += 1
        end
        if norm(M - (U_svd * V_svd).^2) / norm(M) < 1e-3
            nb_good_svd += 1
        end
    end
    
    return [nb_good_rand/nb_tests, nb_good_svd/nb_tests]
end

function random_test(
    extrapoled::Bool = true)
    # Choice of random seed
    Random.seed!(2024)

    file_path = "synthetic data"
    max_iterations = 10000
    alpha = 0.9999
    r = 2
    values = [5, 10, 50, 100]
    nb_tests = 10
    open("results/$(file_path)/accuracy_random_tests_extrapoled=$(extrapoled)_alpha=$(alpha).txt", "w") do file
        for i in values
            println("Matrices nxn : n=",i)
            m = n = i
            nb_good = 0
            U = rand(m, r)
            V = rand(r, n)
            M = (U * V).^2
            for i in 1:nb_tests
                U, V = init_matrix(M, r, "random")
                if extrapoled
                    U, V= coordinate_descent_extrapoled(max_iterations, M, U, V, alpha = alpha)
                else
                    U, V= coordinate_descent(max_iterations, M, U, V, alpha = alpha)
                end
                if norm(M - (U * V).^2) / norm(M) < 1e-3
                    nb_good += 1
                end
            end
            accuracy = nb_good/nb_tests
            write(file, "Matrix mxn : $n\n")
            write(file, "accuracy=$(accuracy * 100)%\n")
        end
    end
end

function multiple_test(
    extrapoled::Bool = true)
    # Choice of random seed
    Random.seed!(2024)
    r = 2
    values = [5, 10, 50, 100]
    alpha = 0.99
    file_path = "synthetic data"
    open("results/$(file_path)/accuracy_multiple_tests_extrapoled=$(extrapoled)_alpha=$(alpha).txt", "w") do file
        for i in values
            println("Matrices nxn : n=",i)
            m = n = i
            accuracy = squared_factorisation(m, n, r, 100, alpha, extrapoled)
            write(file, "Matrix mxn : $n\n")
            write(file, "for random method accuracy=$(accuracy[1] * 100)%\n")
            write(file, "for svd accuracy=$(accuracy[2] * 100)%\n")
        end
    end
end