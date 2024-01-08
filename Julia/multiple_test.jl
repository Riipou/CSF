include("function_CD.jl")

function squared_factorisation(m, n, r, choice,nb_tests)
    max_iterations = 10000
    nb_good = 0
    for i in 1:nb_tests
        U = rand(m, r)
        V = rand(r, n)
        M = (U * V).^2
        U, V = init_matrix(M, r, choice)
        U, V = coordinate_descent(max_iterations, M, U, V)
        if norm(M - (U * V).^2) / norm(M) < 1e-3
            nb_good += 1
        end
    end
    
    return nb_good / nb_tests
end

function multiple_test()
    choice1 = "random"
    choice2 = "SVD"
    r = 2
    values = [5, 10, 50, 100]
    open("../results/accuracy_multiple_tests_julia", "w") do file
        write(file, "RANDOM METHOD :\n")
        for i in values
            println(i)
            m = n = i
            accuracy = squared_factorisation(m, n, r, choice1,1)
            write(file, "m=n=$n accuracy=$(accuracy * 100)%\n")
            if accuracy == 0
                break
            end
        end

        write(file, "SVD METHOD :\n")
        for i in values
            m = n = i
            accuracy = squared_factorisation(m, n, r, choice2,100)
            write(file, "m=n=$n accuracy=$(accuracy * 100)%\n")
            if accuracy == 0
                break
            end
        end
    end
end

multiple_test()