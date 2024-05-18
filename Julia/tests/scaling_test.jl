include("../algorithms/CD.jl")
include("../algorithms/functions.jl")
include("CSFandNMF.jl")
using Random
using Plots

function compare_scaling()
    # Choice of random seed
    Random.seed!(2025)
    open("results/scaling/comparison.txt", "w") do file
        n = 200
        r = 10
        l = 150
        nbr_test=10
        errors_1_m = zeros(l)
        errors_2_m = zeros(l)
        for _ in 1:nbr_test
            U = randn(n,r)
            V = randn(r, n)
            M = (U*V).^2
            U, V = init_matrix(M, r, "random")
            U_1 = randn(size(M, 1), r)
            V_1 = randn(r, size(M, 2))
            errors_1, _ = coordinate_descent_extrapoled(100000, M, U, V, alpha = 0.9999999, errors_calculation = true)
            errors_2, _ = coordinate_descent_extrapoled(100000, M, U_1, V_1, alpha = 0.9999999, errors_calculation = true)
            for i in 1:l
                errors_1_m[i] += errors_1[i]
                errors_2_m[i] += errors_2[i]
            end
        end
        errors_1_m /= nbr_test
        errors_2_m /= nbr_test
        plot(xlabel="Iterations", ylabel="Error", title="Error vs Iterations", legend=true)
        ylims!(0.30, 0.75)
        xlims!(1,l)
        plot!(1:length(errors_1_m), errors_1_m, label="Scaling")
        plot!(1:length(errors_2_m), errors_2_m, label="Unscaled")
        for i in 1:l
            write(file, "($(i), $(errors_1_m[i]))\n")
        end
        for i in 1:l
            write(file, "($(i), $(errors_2_m[i]))\n")
        end
        display(plot!)
        savefig("results/scaling/comparison.png")
    end
end

function time_comparison()
    # Choice of random seed
    Random.seed!(2025)
    # Open data set
    M = open_dataset("CBCL")
    max_time = 70
    open("results/scaling/comparison_time.txt", "w") do file
        r = 20
        U, V = init_matrix(M, r, "random")
        U_1 = randn(size(M, 1), r)
        V_1 = randn(r, size(M, 2))
        errors_1_m, time_1 = coordinate_descent_extrapoled(100000, M, U, V, max_time = max_time, errors_calculation = true)
        errors_2_m, time_2 = coordinate_descent_extrapoled(100000, M, U_1, V_1, max_time = max_time, errors_calculation = true)
        plot(xlabel="Iterations", ylabel="Error", title="Error vs Iterations", legend=true)
        plot!(time_1, errors_1_m, label="Scaling")
        plot!(time_2, errors_2_m, label="Unscaled")
        l1 = length(errors_1_m)
        for i in 1:l1
            write(file, "($(time_1[i]), $(errors_1_m[i]))\n")
        end
        l2 = length(errors_2_m)
        for i in 1:l2
            write(file, "($(time_2[i]), $(errors_2_m[i]))\n")
        end
        display(plot!)
        savefig("results/scaling/comparison_time.png")
    end
end
#= compare_scaling() =#
time_comparison()