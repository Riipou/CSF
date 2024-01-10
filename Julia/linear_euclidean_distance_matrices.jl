include("function_CD.jl")

function euclidean_distance_matrix(n)
    distance_matrix = zeros(n,n)
    for i in 1:n
        for j in 1:n
            distance_matrix[i,j] =(i-j)^2
        end
    end
    return distance_matrix
end

function multiple_distance_matrix()
    nb_tests = 10^4
    max_n = 10
    open("results/distance_matrix_multiple_test.txt", "w") do file
        for i in max_n:-1:3
            println("Matrix of size : ",i)
            write(file, "Matrix : $i x $i\n")
            M = euclidean_distance_matrix(i)
            for r in i-1:-1:1
                println("Rank : ",r)
                nb_good = 0
                best_error = Inf
                for j in 1:nb_tests
                    U, V = init_matrix(M, r, "random")
                    U, V = coordinate_descent(10000, M, U, V)
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
multiple_distance_matrix()