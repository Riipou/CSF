include("../algorithms/CD.jl")

using Random
using Plots
using MAT

function compareCD_extrapolation(M, r)
    open("results/extrapoled_CD/extrapoled_vs_normalCD.txt", "w") do file
        U, V = init_matrix(M, r, "random", p_and_n = false)
        U_1 = copy(U)
        V_1 = copy(V)
        plot(xlabel="Iterations", ylabel="Error", title="Error vs Iterations", legend=true)
        ylims!(0, 0.3)
        xlims!(1,50)
        errors_1 = coordinate_descent(100000, M, U, V, alpha = 0.9999, errors_calculation = true)
        plot!(1:length(errors_1), errors_1, label="CD")
        errors_2 = coordinate_descent_extrapoled(100000, M, U_1, V_1, alpha = 0.9999, errors_calculation = true)
        plot!(1:length(errors_2), errors_2, label="CD extrapoled")
        
        display(plot!)
        savefig("results/extrapoled_CD/error_plot.png")
        write(file, "$(errors_1)\n\n")
        write(file, "$(errors_2)\n\n")
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
# Choice of random seed
Random.seed!(2026)
n = 10
r = 3
U = randn(n,r)
V = randn(r, n)
M = (U*V).^2
#= # Path of data set
file_path = "data sets/CBCLfacialfeatures.mat"
# Load the .mat file
data = matread(file_path)
# Access variables in the .mat file
M = data["U"] =#
compareCD_extrapolation(M, r)
