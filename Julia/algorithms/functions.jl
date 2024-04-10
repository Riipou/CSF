function init_matrix(
    M,
    r,
    choice;
    p_and_n::Bool = true)
    
    if choice == "random"
        if p_and_n
            U = randn(size(M, 1), r)
            V = randn(r, size(M, 2))
            A = (U*V).^2
            lambda = sum(A[:].*M[:])/sum(A[:].^2)
            if lambda < 0
                U = rand(size(M, 1), r)
                V = rand(r, size(M, 2))
                A = (U*V).^2
                lambda = sum(A[:].*M[:])/sum(A[:].^2)
            end
        else 
            U = rand(size(M, 1), r)
            V = rand(r, size(M, 2))
            A = (U*V).^2
            lambda = sum(A[:].*M[:])/sum(A[:].^2)
        end
        U *= lambda^(1/4)
        V *= lambda^(1/4)
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