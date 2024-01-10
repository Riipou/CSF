include("function_CD.jl")

function squared_factorisation(m, n, r,nb_tests)
    max_iterations = 10000
    nb_good_rand = 0
    nb_good_svd = 0
    for i in 1:nb_tests
        U = rand(m, r)
        V = rand(r, n)
        M = (U * V).^2
        U_rand, V_rand = init_matrix(M, r, "random")
        U_svd, V_svd = init_matrix(M, r, "SVD")
        U_rand, V_rand = coordinate_descent(max_iterations, M, U_rand, V_rand)
        U_svd, V_svd = coordinate_descent(max_iterations, M, U_svd, V_svd)
        if norm(M - (U_rand * V_rand).^2) / norm(M) < 1e-3
            nb_good_rand += 1
        end
        if norm(M - (U_svd * V_svd).^2) / norm(M) < 1e-3
            nb_good_svd += 1
        end
    end
    
    return [nb_good_rand / nb_tests,nb_good_svd/nb_tests]
end

function random_test()
    open("results/accuracy_random_tests_julia.txt", "w") do file
        r = 2
        values = [5, 10, 50, 100]
        for i in values
            println("Matrices nxn : n=",i)
            m = n = i
            nb_tests = 10
            max_iterations = 10000
            nb_good = 0
            U = rand(m, r)
            V = rand(r, n)
            M = (U * V).^2
            for i in 1:nb_tests
                U, V = init_matrix(M, r, "random")
                U, V= coordinate_descent(max_iterations, M, U, V)
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

function multiple_test()
    r = 2
    values = [5, 10, 50, 100]
    open("results/accuracy_multiple_tests_julia.txt", "w") do file
        for i in values
            println("Matrices nxn : n=",i)
            m = n = i
            accuracy = squared_factorisation(m, n, r,100)
            write(file, "Matrix mxn : $n\n")
            write(file, "for random method accuracy=$(accuracy[1] * 100)%\n")
            write(file, "for svd accuracy=$(accuracy[2] * 100)%\n")
        end
    end
end

random_test()