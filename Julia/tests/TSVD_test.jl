include("CSFandNMF.jl")

using Random
using TSVD

function tsvd_datasets()
    datasets = ["CBCL","TDT2","CBCLfacialfeatures"]
    for d in datasets
        open("results/tsvd/$(d).txt", "w") do file
            Random.seed!(2024)
            if d == "CBCL"
                r_values = [10,20,49]
            else
                r_values = [10,20,30,40]
            end
            m = n = 10
            for r in r_values
                # Open data set
                M = open_dataset(d)
                U, sigma, Vt = tsvd(M, r)
                sigma = Diagonal(sigma)
                write(file,"Error r = $(r): \n")
                write(file,"$(norm(M-(U*sigma*Vt'))/norm(M))\n")
            end
        end
    end
end

function tsvd_sparse_matrices(
    nb_tests::Int,
    size::Int,
    r::Int,
    density::Float64)
    # Choice of random seed
    Random.seed!(2024)


    # Definition of variables
    errors = []

    println("Size : ", size)
    
    for i in 1:nb_tests

        println(i)

        # Generate a sparse matrix of size size-by-size with density = density
        M = sprand(size, size, density)

        # Ensure at least one non-zero element per row and per column
        added = 0
        for i in 1:size
            if nnz(M[i, :]) == 0
                j = rand(1:size)
                M[i, j] = rand()
                added += 1 
            end
        end

        for j in 1:size
            if nnz(M[:, j]) == 0
                i = rand(1:size)
                M[i, j] = rand()
                added += 1 
            end
        end
        for i in 1:size
            if nnz(M[i, :]) > 1 && added > 0      
                indices, _ = findnz(M[i,:])
                for index in indices
                    if nnz(M[:, index]) > 1 && nnz(M[i, :]) > 1 && added > 0
                        M[i,index] = 0
                        added -= 1
                    end
                end 
            end
            if nnz(M[:, i]) > 1 && added > 0      
                indices, _ = findnz(M[i,:])
                for index in indices
                    if nnz(M[index, :]) > 1 && nnz(M[:, i]) > 1 && added > 0
                        M[index,i] = 0
                        added -= 1
                    end
                end 
            end
        end
        M = Matrix(M)
        # TSVD
        U, sigma, Vt = tsvd(M, r)
        sigma = Diagonal(sigma)
        push!(errors, norm(M-(U*sigma*Vt'))/norm(M))
    end

    return errors
end 

function sparse_rank_tests()
    size = 200
    r_values = [10, 20]
    density = 0.05
    nb_tests = 10
    open("results/tsvd/sparse_matrices_$(size)x$(size)_rank_test.txt", "w") do file
        for r in r_values
            errors = tsvd_sparse_matrices(nb_tests, size, r, density)
            write(file,"r = $(r)\n")
            write(file,"error = $((sum(errors)/nb_tests) * 100)%\n")
            write(file, "standard deviation = $(std(errors)*100)%\n")
        end
    end
end

function sparse_size_tests()
    size_list = [100, 250, 500, 750, 1000]
    r = 10
    density = 0.05
    nb_tests = 10
    open("results/tsvd/sparse_matrices_size_test.txt", "w") do file
        for size in size_list
            errors = tsvd_sparse_matrices(nb_tests, size, r, density)
            write(file,"size = $(size)x$(size)\n")
            write(file,"error = $((sum(errors)/nb_tests) * 100)%\n")
            write(file, "standard deviation = $(std(errors)*100)%\n")
        end
    end
end

function sparse_density_tests()
    density_list = [ 0.50, 0.40, 0.30, 0.20, 0.10, 0.01]
    r = 10
    size = 200
    nb_tests = 10
    open("results/tsvd/sparse_matrices_density_test.txt", "w") do file
        for density in density_list
            errors = tsvd_sparse_matrices(nb_tests, size, r, density)
            write(file,"size = $(density)\n")
            write(file,"error = $((sum(errors)/nb_tests) * 100)%\n")
            write(file, "standard deviation = $(std(errors)*100)%\n")
        end
    end
end