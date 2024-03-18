include("../algorithms/CD.jl")
include("../algorithms/functions.jl")

function BLS(
    max_iterations::Int,
    M::Matrix,
    U::Matrix,
    V::Matrix)

    m, r = size(U)
    r, n = size(V)
    alpha_U = fill(1, m)
    alpha_V = fill(1, n)
    for _ in max_iterations
        for j in n
            V, alpha_V[j] = update_V(alpha_V[j], U, V, M, j)
        end
        # for i in m
        #     U[i,:], alpha_U[i] = update_x(alpha_U[i], V', U[i,:]', M[i,:]')'
        # end
    end

end

function update_V(alpha, U, V, M, j)
    loss = loss_function(M, U, V)
    alpha *= 1.5
    X = copy(V)
    X[:,j] = V[:,j] - alpha * grad(U, V[:,j], M[:,j])
    while loss_function(M, U, X) > loss
        alpha /= 2
        X[:,j] = V[:,j] - alpha * grad(U, V[:,j], M[:,j])
    end
    return X, alpha
end

function loss_function(M, U, V)
    return norm(M - (U * V).^2)
end

function grad(A, x, b)
    return 4*A'*(((A*x).^2-b).*(A*x))
end

U = [1 2 3; 4 5 6; 7 8 9]

V = [9 6 2 4; 4 2 1 3; 4 5 6 7]

M = (U * V).^2

U, V = init_matrix(M, 3, "random")

BLS(10000, M, U, V)