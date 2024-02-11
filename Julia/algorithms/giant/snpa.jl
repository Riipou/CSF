export snpa

function snpa(X::AbstractMatrix{T},
              r::Integer,
              epsilon::Float64=10e-9;
              normalize::Bool=false) where {T<:AbstractFloat}
    # Get dimensions
    m, n = size(X)

    # Optionally normalize so that columns of X sum to one
    if normalize
        X = copy(X) # copy in local scope to avoid modifying input
        for col in eachcol(X)
            col ./= sum(col)
        end
    end

    # Init
    # Set of selected indices
    K = zeros(Int, r)
    # Norm of columns of input X
    normX0 = sum.(abs2,eachcol(X))
    # Max of the columns norm
    nXmax = maximum(normX0)
    # Init residual
    normR = copy(normX0)
    # Init set of extracted columns
    U = zeros(T, m, r)
    # Init output H
    H = zeros(T, r, n)

    # Update intermediary variables (save time for norm computations)
    XtUK = zeros(T, n, r) # X^T * U(:,K)
    UKtUK = zeros(T, r, r) # U(:,K)^T * U(:,K)

    # SNPA loop
    i = 1
    while i <= r && sqrt(maximum(normR)/nXmax) > epsilon
        # Select column of X with largest l2-norm
        a = maximum(normR)
        # Check ties up to 1e-6 precision
        b = findall((a .- normR) / a .<= 1e-6)
        # In case of a tie, select column with largest norm of the input matrix
        _, d = findmax(normX0[b])
        b = b[d]
        # Save index of selected column, and column itself
        K[i] = b
        U[:,i] .= X[:,b]
        # Update XtUK
        XtUK[:,i] .= X'*U[:,i]
        # Update UKtUK
        if i == 1
            UKtUK[1,1] = U[:,i]' * U[:,i]
        else
            UtUi = U[:,1:i-1]' * U[:,i]
            UKtUK[1:i-1,i] .= UtUi
            UKtUK[i,1:i-1] .= UtUi
            UKtUK[i,i] = U[:,i]' * U[:,i]
        end
        # Update residual, that is solve
        # min_Y ||X - X[:,J] Y||
        # TODO add constraint that Y^T * e <= e with e vec of ones
        H[1:i,:] .= matrixactiveset(X, X[:,K[1:i]], H[1:i,:])
        # Update the norm of the columns of the residual
        normR .= (normX0
                  - 2 * sum.(eachcol(XtUK[:,1:i]' .* H[1:i,:]))
                  + sum.(eachcol(H[1:i,:] .* (UKtUK[1:i,1:i] * H[1:i,:]))))
        # Increment iterator
        i += 1
    end

    return K
end
