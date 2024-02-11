using LinearAlgebra
using Random

export randspa, randorth!

# Randomized SPA
function randspa(X::AbstractMatrix{T},
                 r::Integer;
                 d::Integer=1,
                 genP::Function=randn!, # Function to generate random P
                 Prank::Int=0
                 ) where T <: AbstractFloat
    # Get dimensions
    m, n = size(X)

    # Set of selected indices
    K = zeros(Int, r)
    # Norm of columns of input X
    normX0 = sum.(x->x^2,eachcol(X))

    # Init
    if iszero(Prank) Prank = min(2*r, m) end
    P = Matrix{T}(undef, Prank, m)
    U = zeros(T, m, r)
    V = zeros(T, r, n)
    PU = zeros(T, Prank, r)
    PUVj = Vector{T}(undef, Prank)
    PXj = Vector{T}(undef, Prank)
    QnormR = Vector{T}(undef, n)
    dQnormR = Vector{T}(undef, d)
    dK = zeros(Int, d)

    for i in 1:r
        for k in 1:d
            genP(P)
            if i>1 mul!(PU, P, U) end
            @views for j in eachindex(QnormR)
                mul!(PXj, P, X[:,j])
                # no need to compute specifically mul!(PUVj,PU[:,1:i-1],V[1:i-1,j]) since PU and V are intialized with zeros
                mul!(PUVj, PU, V[:,j])
                QnormR[j] = sum(abs2, PXj) + sum(abs2, PUVj) - 2*PXj'PUVj
            end
            # Select column of X with largest l2-norm
            a = maximum(QnormR)
            # Check ties up to 1e-6 precision
            b = findall((a .- QnormR) / a .<= 1e-6)
            # In case of a tie, select column with largest norm of the input matrix
            bmax = argmax(normX0[b])
            b = b[bmax]
            # Save index of selected column
            dK[k] = b
            if d>1
                update_UV(U, V, X, i, b)
                dQnormR[k] = sum(abs2, V[i,:])
                U[:,i] .= zero(T)
                V[i,:] .= zero(T)
            end
        end
        b = dK[argmax(dQnormR)]
        K[i] = b
        # Update U and V
        update_UV(U, V, X, i, b)
    end
    return K
end


function update_UV(U, V, X, i, b)
    @views begin
        U[:,i] .= X[:,b]
        for j in 1:i-1
            U[:,i] .-= U[:,j] * (U[:,j]' * U[:,i])
        end
    normalize!(U[:,i])
    mul!(V[i,:], X', U[:,i])
    end
end


# Generate a random matrix with orthogonal columns and condition number κ
# To run randSPA like VCA:
# g = x->randorth!(x, Inf); randspa(X, r, genP=g)
# To run randSPA like SPA:
# g = x->randorth!(x, 1); randspa(X, r, genP=g)
# Pick an intermediary κ for a balanced randSPA, eg κ=1.5
function randorth!(P, κ)
    if isinf(κ)
        randn!(P)
        P[2:end,:] .= 0
    elseif κ == 1
        P .= 0
        P[diagind(P)].=1
    else
        randn!(P)
        copy!(P, Matrix(qr(P').Q)[:,1:size(P,1)]')
        rootκ = sqrt(κ)
        P[2:end,:] ./= rootκ
    end
end
