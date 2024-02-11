using LinearAlgebra

export vca

function vca(X::AbstractMatrix{T},
             r::Integer;
             epsilon::Float64=10e-9,
             ) where T <: AbstractFloat
    # Get dimensions
    m, _ = size(X)

    # Set of selected indices
    K = zeros(Int, r)

    # Norm of columns of input X
    normX0 = sum.(abs2,eachcol(X))
    # Max of the columns norm
    nXmax = maximum(normX0)
    # Init residual
    normR = copy(normX0)
    V = zeros(T, m, r)
    W = zeros(T, m, r)

    i = 1
    while i <= r && sqrt(maximum(normR)/nXmax) > epsilon
        # Select random direction in col(Y)
        diru = randn(T, m)
        # Project the direction, to be orthogonal to previously extracted cols
        if i > 1
            diru = diru - V*(V'*diru)
        end
        # Find the data point (col of X) the most correlated with that dir
        u = (diru'*X)'
        a = maximum(abs.(u))
        # Check ties up to 1e-6 precision
        b = findall((a .- abs.(u)) / a .<= 1e-6)
        # In case of a tie, select column with largest norm of the input matrix
        _, d = findmax(normX0[b])
        b = b[d]
        # Save index of selected column
        K[i] = b
        # Update W
        W[:,i] .= X[:, K[i]]
        # Update projector
        updateorthbasis!(V, W[:,i], i)
        # Increment iterator
        i += 1
    end

    return K
end


# Update the orthogonal basis on which we project
function updateorthbasis!(V::AbstractMatrix{T},
                          vec::AbstractVector{T},
                          idx::Int
                          ) where T <: AbstractFloat
    if iszero(V)
        V[:,1] = vec / norm(vec)
    else
        # Project new vector onto orthogonal complement and normalize
        vec = vec - V*(V'*vec)
        vec = vec / norm(vec)
        V[:,idx] .= vec
    end
end
