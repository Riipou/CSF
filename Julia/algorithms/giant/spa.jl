using LinearAlgebra

export spa

function spa(X::AbstractMatrix{T},
             r::Integer,
             epsilon::Float64=10e-9) where T <: AbstractFloat
    # Get dimensions
    m, n = size(X)

    # Set of selected indices
    K = zeros(Int, r)

    # Norm of columns of input X
    normX0 = sum.(abs2,eachcol(X))
    # Max of the columns norm
    nXmax = maximum(normX0)
    # Init residual
    normR = copy(normX0)
    # Init set of extracted columns
    U = Matrix{T}(undef,m,r)

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
        # Update residual
        for j in 1:i-1
            U[:,i] .= U[:,i] - U[:,j] * (U[:,j]' * U[:,i])
        end
        U[:,i] ./= norm(U[:,i])
        normR .-= (X'*U[:,i]).^2
        # Increment iterator
        i += 1
    end

    return K
end
