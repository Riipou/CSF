include("../algorithms/CD.jl")
include("../algorithms/functions.jl")

using Random

function multiple_slackgon_matrix()
    
    # Choice of random seed
    Random.seed!(2024)

    nb_tests = 10^3
    max_n = 10
    alpha = 0.99999
    file_path = "slack matrices of n-gons"
    open("results/$(file_path)/slackgon_matrix_aplha=$(alpha).txt", "w") do file
        for i in max_n:-1:3
            println("Matrix of size : ",i)
            write(file, "Matrix : $i x $i\n")
            M, _ = slackngon(i)
            for r in i-1:-1:1
                println("Rank : ",r)
                nb_good = 0
                best_error = Inf
                for _ in 1:nb_tests
                    U, V = init_matrix(M, r, "random")
                    U, V = coordinate_descent(10000, M, U, V, alpha = alpha)
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