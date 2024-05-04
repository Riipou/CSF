using Random
include("../algorithms/functions.jl")
include("../algorithms/GD.jl")
include("CSFandNMF.jl")
include("../algorithms/slackgon.jl")

function GDandCD(
    max_time::Int,
    r::Int,
    nb_tests::Int,
    dataset::String;
    max_iterations::Int = 100000,
    p_and_n::Bool = true)
    
    # Choice of random seed
    Random.seed!(2024)

    # Open data set
    M = open_dataset(dataset)

    open("results/GDandCD/$(dataset)_r=$(r)_max_time=$(max_time)_p&n=$(p_and_n).txt", "w") do file

        # Definition of variables
        errors_GD = []
        errors_CD = []

        m,n = size(M)
        U = zeros(m,r)
        V = zeros(r,n)
        best_U_CD = copy(U)
        best_V_CD = copy(V)
        
        best_U_GD = zeros(m,r)
        best_V_GD = zeros(r,n)
        # CSF loop
        for i in 1:nb_tests
            println(i)
            U_0, V_0 = init_matrix(M, r, "random", p_and_n = p_and_n)
            # CD
            U, V = coordinate_descent_extrapoled(max_iterations, M, U_0, V_0, max_time = max_time)
            push!(errors_CD, norm(M - (U * V).^2)/norm(M))
            if  norm(M - (U*V).^2)/norm(M) <  norm(M - (best_U_CD * best_V_CD).^2)/norm(M)
                best_U_GD = U
                best_V_GD = V
            end
            # GD
            U, V = BLS(max_iterations, M, U_0, V_0, max_time = max_time)
            push!(errors_GD, norm(M - (U * V).^2)/norm(M))
            if  norm(M - (U*V).^2)/norm(M) <  norm(M - (best_U_GD * best_V_GD).^2)/norm(M)
                best_U_GD = U
                best_V_GD = V
            end
        end

        average_error_CD = sum(errors_CD)/nb_tests
        average_error_GD = sum(errors_GD)/nb_tests

        # Results
        write(file, "CD\n")
        write(file, "average error = $(average_error_CD * 100)%\n")
        write(file, "standard deviation = $(std(errors_CD)*100)%\n")
        write(file, "GD\n")
        write(file, "average error = $(average_error_GD * 100)%\n")
        write(file, "standard deviation = $(std(errors_GD)*100)%\n")
        
        # Create a dictionary containing the matrices
        data_dict = Dict("bUCD" => best_U_CD, "bVCD" => best_V_CD,"MCD" => (best_U_CD * best_V_CD).^2,"bUGD" => best_U_GD, "bVGD" => best_V_GD,"MGD" => (best_U_GD * best_V_GD).^2)
        # Save the dictionary to a .mat file
        matwrite("results/GDandCD/matrices/$(dataset)_r=$(r)_max_time=$(max_time)_p&n=$(p_and_n).mat", data_dict)

        end
end

function extrapolation_parameters_GD()

    beta_bis_list = [ 0.15, 0.3, 0.5, 0.75]
    gamma_list = [1.01, 1.05, 1.1]
    gamma_bis_list = [1.005, 1.01, 1.05] 
    eta_list = [1.5,2.0,3.0]
    nb_tests = 10
    l = 150
    gamma = 1.1
    gamma_bis = 1.05
    beta_bis = 0.5
    r = 10
    eta = 1.5
    plot(xlabel="Iterations", ylabel="Error", title="Error vs Iterations", legend=true)
    ylims!(0.04, 0.07)
    xlims!(1,l)
    for beta_bis in beta_bis_list
        #= if gamma == 1.01
            gamma_bis = 1.005
        elseif gamma == 1.05
            gamma_bis = 1.01
        else
            gamma_bis = 1.05
        end =#
        Random.seed!(2026)
        U = randn(200,r)
        V = randn(r,200)
        M = (U*V).^2
        errors_2_m = zeros(l)
        open("results/extrapoled_GD/parameters/extrapolation_parameters_beta=$(beta_bis)_gamma=$(gamma)_gammabis=$(gamma_bis)_eta=$(eta).txt", "w") do file
            for _ in nb_tests
                U, V = init_matrix(M, r, "random", p_and_n = false)
                U_1 = copy(U)
                V_1 = copy(V)
                errors_2, _ = BLS(100000, M, U_1, V_1, alpha = 0.9999, errors_calculation = true, beta_bis = beta_bis, eta = eta, gamma = gamma, gamma_bis = gamma_bis)
                for i in 1:l
                    errors_2_m[i] += errors_2[i]
                end
            end
            errors_2_m /= nb_tests
            write(file, "Error GD :\n")
            for i in 1:l
                write(file, "($(i),$(errors_2_m[i]))\n")
            end
            plot!(1:length(errors_2_m), errors_2_m, label="beta=$(beta_bis)_gamma=$(gamma)_gammabis=$(gamma_bis)_eta=$(eta)")
            display(plot!)
        end
    end
    savefig("results/extrapoled_GD/parameters/extrapolation_parameters.png")
end

function GDandCD_comparison(
    alpha::Float64,
    r::Int,
    dataset::String;
    max_iterations::Int = 100000,
    p_and_n::Bool = true)
    
    # Choice of random seed
    Random.seed!(2024)

    # Open data set
    M = open_dataset(dataset)

    l = 150
    plot(xlabel="Time", ylabel="Error", title="Error vs Time", legend=true)
    ylims!(0.0, 1.0)
    xlims!(0.0, 60.0)

    #= m=n=1000
    U = randn(m, r)
    V = randn(r, n)
    M = (U * V).^2 =#

    open("results/GDandCD/$(dataset)_comparison_r=$(r)_p&n=$(p_and_n).txt", "w") do file
        write(file, "Error CD :\n")
        for i in 1:l
            write(file, "($(times_CD[i]),$(errors_CD[i]))\n")
        end
        plot!(times_CD, errors_CD, label="CD")
        display(plot!)
        write(file, "Error GD :\n")
        for i in 1:l
            write(file, "($(times_GD[i]),$(errors_GD[i]))\n")
        end
        plot!(times_GD, errors_GD, label="GD")
        display(plot!)
    end    
    savefig("results/GDandCD/comparison3.png")
end

function multipleGD_slackgon_matrix()
    
    # Choice of random seed
    Random.seed!(2024)

    nb_tests = 10^3
    max_n = 10
    alpha = 0.9999
    open("results/GDandCD/slackgon_matrix_aplha=$(alpha)_n=12-9.txt", "w") do file
        for i in max_n:-1:3
            println("Matrix of size : ",i)
            write(file, "Matrix : $i x $i\n")
            M, _ = slackngon(i)
            for r in i-1:-1:i-3
                println("Rank : ",r)
                nb_good = 0
                best_error = Inf
                best_U = zeros(i,r)
                best_V = zeros(r,i)
                for _ in 1:nb_tests
                    U, V = init_matrix(M, r, "random")
                    BLS(100000, M, U, V, alpha = alpha)
                    error = norm(M - (U * V).^2) / norm(M)
                    if error < norm(M - (best_U * best_V).^2) / norm(M)
                        best_U = copy(U)
                        best_V = copy(V)
                    end
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
                #= if accuracy == 0
                    break
                end =#
            end
        end
    end
end

GDandCD_comparison(100.0, 20, "CBCL", p_and_n = false)