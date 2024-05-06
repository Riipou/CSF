using LinearAlgebra

function BLS_nonextrapoled(
    max_iterations::Int,
    M::Matrix,
    U::Matrix,
    V::Matrix;
    step::Float64 = 1.1,
    max_time:: Int = 0,
    alpha:: Float64 = Inf,
    errors_calculation::Bool = false)

    if max_time == 0
        max_time = Inf
    end

    if errors_calculation
        errors = []
        times = []
    end

    m, r = size(U)
    r, n = size(V)
    U = convert(Matrix{Float64}, U)
    V = convert(Matrix{Float64}, V)
    if alpha < 1
        prev_error = norm(M - (U * V).^2)
    end
    # Define alpha lists
    alpha_V = []
    alpha_U = []

    # Compute alpha lists
    for j in 1:n
        push!(alpha_V, step * (norm(V[:,j])/norm(grad(U, V[:,j], M[:,j]))))
    end
    for i in 1:m
        push!(alpha_U, step * (norm(U[i,:])/norm(grad(V', U[i,:], M[i,:]))))
    end

    start = time()
    # GD
    for ite in 1:max_iterations
        for j in 1:n
            V[:,j], alpha_V[j] = update_x(alpha_V[j], U, V[:,j], M[:,j])
        end
        for i in 1:m
            U[i,:], alpha_U[i] = update_x(alpha_U[i], V', U[i,:], M[i,:])
        end

        error = norm(M - (U * V).^2)

        if errors_calculation
            push!(errors, norm(M - (U * V).^2)/norm(M))
            push!(times, time()-start)
        end

        if ite % 10 == 0 && alpha <= 1
            if error > alpha * prev_error
                break
            end
            prev_error = norm(M - (U * V).^2)
        end

        if time()-start >= max_time
            break
        end 
    end

    if errors_calculation
        return errors, times
    else
        return U, V
    end
end

function BLS(
    max_iterations::Int,
    M::Matrix,
    U::Matrix,
    V::Matrix;
    step::Float64 = 1.1,
    max_time:: Int = 0,
    alpha:: Float64 = Inf,
    beta_bis::Float64 = 0.5,
    eta::Float64 = 1.5,
    gamma::Float64 = 1.1,
    gamma_bis::Float64 = 1.05,
    errors_calculation::Bool = false)

    if max_time == 0
        max_time = Inf
    end

    if errors_calculation
        errors = []
        times = []
    end

    m, r = size(U)
    r, n = size(V)
    U = convert(Matrix{Float64}, U)
    V = convert(Matrix{Float64}, V)
    if alpha < 1
        prev_error = norm(M - (U * V).^2)
    end
    # Define alpha lists
    alpha_V = []
    alpha_U = []
    # Parameters of extrapoled GD
    beta = 1
    beta_pred = beta_bis
    Up = copy(U)
    Vp = copy(V)
    Vtemp = copy(V)
    Utemp = copy(U)
    # Compute alpha lists
    for j in 1:n
        push!(alpha_V, step * (norm(V[:,j])/norm(grad(U, V[:,j], M[:,j]))))
    end
    for i in 1:m
        push!(alpha_U, step * (norm(U[i,:])/norm(grad(V', U[i,:], M[i,:]))))
    end

    start = time()
    # Extrapoled GD
    for ite in 1:max_iterations
        alpha_U_bis = copy(alpha_U)
        alpha_V_bis = copy(alpha_V)
        for j in 1:n
            V[:,j], alpha_V[j] = update_x(alpha_V[j], U, V[:,j] + beta_bis * (V[:,j] - Vp[:,j]), M[:,j])
        end
        Vp = copy(Vtemp)
        for i in 1:m
            U[i,:], alpha_U[i] = update_x(alpha_U[i], V', U[i,:] + beta_bis * (U[i,:] - Up[i,:]), M[i,:])
        end
        Up = copy(Utemp)
        Utemp = copy(U)
        Vtemp = copy(V)

        error = norm(M - (U * V).^2)

        if error > norm(M - (Up*Vp).^2)
            for j in 1:n
                V[:,j], alpha_V[j] = update_x(alpha_V_bis[j], Up, Vp[:,j], M[:,j])
            end
            for i in 1:m
                U[i,:], alpha_U[i] = update_x(alpha_U_bis[i], Vp', Up[i,:], M[i,:])
            end
            Utemp = copy(U)
            Vtemp = copy(V)
            error = norm(M - (U*V).^2)
            beta_copy = copy(beta_bis)
            beta_bis = beta_bis/eta
            beta = beta_pred
            beta_pred = beta_copy
        else
            beta_pred = beta_bis
            beta_bis = min(beta, gamma * beta_bis)
            beta = min(1, gamma_bis * beta)
        end

        if errors_calculation
            push!(errors, norm(M - (U * V).^2)/norm(M))
            push!(times, time()-start)
        end

        if ite % 10 == 0 && alpha <= 1
            if error > alpha * prev_error
                break
            end
            prev_error = norm(M - (U * V).^2)
        end

        if time()-start >= max_time
            break
        end 
    end

    if errors_calculation
        return errors, times
    else
        return U, V
    end
end

function update_x(alpha, A, x, b)
    loss = loss_function(A, x, b)
    alpha *= 1.5
    x2 = copy(x)
    x2 = x - alpha * grad(A, x, b)
    while loss_function(A, x2, b) > loss
        alpha /= 2
        x2 = x - alpha * grad(A, x, b)
    end
    return x2, alpha
end

function loss_function(A, x, b)
    return norm((A * x).^2 - b)/norm(b)
end

function grad(A, x, b)
    return 4*A'*(((A*x).^2-b).*(A*x))
end