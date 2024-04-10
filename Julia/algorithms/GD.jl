function BLS(
    max_iterations::Int,
    M::Matrix,
    U::Matrix,
    V::Matrix)

    m, r = size(U)
    r, n = size(V)
    U = convert(Matrix{Float64}, U)
    V = convert(Matrix{Float64}, V)
    alpha_V = []
    alpha_U = []
    step = 1.1
    for j in 1:n
        push!(alpha_V, step * (norm(V[:,j])/norm(grad(U, V[:,j], M[:,j]))))
    end
    for i in 1:m
        push!(alpha_U, step * (norm(U[i,:])/norm(grad(V', U[i,:], M[i,:]))))
    end
    for _ in 1:max_iterations
        for j in 1:n
            V[:,j], alpha_V[j] = update_x(alpha_V[j], U, V[:,j], M[:,j])
        end
        for i in 1:m
            U[i,:], alpha_U[i] = update_x(alpha_U[i], V', U[i,:], M[i,:])
        end
    end

    return U, V
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