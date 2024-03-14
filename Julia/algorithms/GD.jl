include("../algorithms/CD.jl")
include("../algorithms/functions.jl")

function BLS(
    max_iterations::Int,
    M::Matrix,
    U::Matrix,
    V::Matrix)

    alpha = 1

    r, n = size(V)
    for _ in max_iterations
        for j in n
            grad(U, V[])
        end
    end

end

function grad(A, x, b)
    return 4*A'*(((A*x).^2-b).*(A*x))
end

U = [1 2 3; 4 5 6; 7 8 9]

V = [9 6 2 4; 4 2 1 3; 4 5 6 7]

M = (U * V).^2

U, V = init_matrix(M, 3, "random")

W = grad(U, V[:,1], M[:,1])
print(W)