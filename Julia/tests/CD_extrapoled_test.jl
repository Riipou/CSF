include("../algorithms/CD.jl")

using Random
using Plots
using MAT
using Dates

function compareCD_extrapolation(M, r)
    open("results/extrapoled_CD/extrapoled_vs_normalCD.txt", "w") do file
        U, V = init_matrix(M, r, "random", p_and_n = false)
        U_1 = copy(U)
        V_1 = copy(V)
        plot(xlabel="Iterations", ylabel="Error", title="Error vs Iterations", legend=true)
        ylims!(0.4, 0.8)
        xlims!(1,100)
        errors_1 = coordinate_descent(100000, M, U, V, alpha = 0.99999, errors_calculation = true)
        plot!(1:length(errors_1), errors_1, label="CD")
        errors_2 = coordinate_descent_extrapoled(100000, M, U_1, V_1, alpha = 0.99999, errors_calculation = true)
        plot!(1:length(errors_2), errors_2, label="CD extrapoled")
        display(plot!)
        savefig("results/extrapoled_CD/error_plot.png")
        write(file, "$(errors_1)\n\n")
        write(file, "$(errors_2)\n\n")
        return errors_1, errors_2
    end
end

function extrapolation_parameters(M)

    beta_bis = 0.5
    gamma = 1.05
    gamma_bis = 1.01
    eta = 1.5

    open("results/extrapoled_CD/extrapolation_parameters_beta=$(beta_bis)_gamma=$(gamma)_gammabis=$(gamma_bis)_eta=$(eta).txt", "w") do file
        U, V = init_matrix(M, r, "random", p_and_n = false)
        U_1 = copy(U)
        V_1 = copy(V)
        plot(xlabel="Iterations", ylabel="Error", title="Error vs Iterations", legend=true)
        ylims!(0, 1.0)
        xlims!(1,50)
        errors_1 = coordinate_descent(100000, M, U, V, alpha = 0.9999, errors_calculation = true)
        plot!(1:length(errors_1), errors_1, label="CD")
        errors_2 = coordinate_descent_extrapoled(100000, M, U_1, V_1, alpha = 0.9999, errors_calculation = true)
        plot!(1:length(errors_2), errors_2, label="CD extrapoled")
        
        display(plot!)
        savefig("results/extrapoled_CD/extrapolation_parameters.png")
        write(file, "$(errors_1)\n\n")
        write(file, "$(errors_2)\n\n")
    end
end

function extrapoledCD_test()
    # Choice of random seed
    Random.seed!(2025)
    open("results/extrapoled_CD/extrapoled_vs_normalCD_errors.txt", "w") do file
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
            errors_1, errors_2 = compareCD_extrapolation(M, r)
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
        plot!(1:length(errors_1_m), errors_1_m, label="CD")
        plot!(1:length(errors_2_m), errors_2_m, label="CD extrapoled")
        write(file, "$(errors_1_m)\n\n")
        write(file, "$(errors_2_m)\n\n")
        display(plot!)
        savefig("results/extrapoled_CD/mean_error_plot.png")
    end
end