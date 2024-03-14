function init_matrix(M, r, choice)
    if choice == "random"
        U = rand(size(M, 1), r)
        V = rand(r, size(M, 2))
        A = (U*V).^2
        lambda = sum(A[:].*M[:])/sum(A[:].^2)
        U *= lambda
        V *= lambda
    elseif choice == "SVD"
        U, S, V = svd(M)
        V = V'
        U = U[:, 1:r]
        V = V[1:r, :]
        U = U * Diagonal(sqrt.(S[1:r]))
        V = Diagonal(sqrt.(S[1:r])) * V
    end
    return U, V
end