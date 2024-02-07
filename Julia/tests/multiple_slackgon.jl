include("../algorithms/function_CD.jl")
include("../algorithms/slackgon.jl")
function multiple_slackgon_matrix()
    nb_tests = 10^3
    max_n = 10
    alpha = 0.999999
    open("results/slackgon_matrix_multiple_test_aplha_0.999999.txt", "w") do file
        for i in max_n:-1:3
            println("Matrix of size : ",i)
            write(file, "Matrix : $i x $i\n")
            M, rank = slackngon(i)
            for r in i-1:-1:1
                println("Rank : ",r)
                nb_good = 0
                best_error = Inf
                for j in 1:nb_tests
                    U, V = init_matrix(M, r, "random")
                    U, V = coordinate_descent(10000, M, U, V, alpha)
                    error = norm(M - (U * V).^2) / norm(M)
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
                if accuracy == 0
                    break
                end
            end
        end
    end
end

multiple_slackgon_matrix()