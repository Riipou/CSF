using SparseArrays

export beta_updtH
export beta_updtW

# Return numerator and denominateur for MU-Beta-NMF
# readmeBetaNMF.md for more explanations !
#
# Input : 
#
# X : n x m matrix
# W : m x r matrix
# H : r x n matrix
# beta : parameter for beta-divergence (Float64)
# fun : Function to apply
# epsilon : tolerence to avoid zero-blocking

# Output :
# N = Numerator of multiplicative update
# D = Denominator of multiplicative update

function num_denom(X::AbstractMatrix{T},
                   W::AbstractMatrix{T},
                   H::AbstractMatrix{T},
                   beta::Float64=1.0,
                   fun::Function=xdy,
                   epsilon::Float64=eps();
                   args...
                   ) where T <: AbstractFloat

    if beta == 1.0
        if issparse(X)
            XdWH = blockrecursivecompwiseprodsparselowrank(X,W,H,fun) 
        else
            XdWH = X./(W*H .+ epsilon)
        end 
        N = W'*XdWH 
        D = repeat(sum(W,dims = 1)',1,size(X,2)) .+ epsilon

    elseif beta == 2.0
        N = W'*X
        WtW = W'*W 
        D = WtW*H .+ epsilon
    
    else
        WH = W*H .+ epsilon 
        N = W'*((WH .+ epsilon).^(beta-2) .* X)
        D = W'*((WH .+ epsilon).^(beta-1))
    end

    return N, D
end 

# Update W by applying multiplicative update using N and D
# Multiplicative factor availible on markdown file

function beta_updtW(X::AbstractMatrix{T},
                    W::AbstractMatrix{T},
                    Ht::AbstractMatrix{T};
                    beta::Float64=1.0,
                    epsilon::Float64=eps(),
                    args...
                    ) where T <: AbstractFloat

    N, D = num_denom(X',Ht,W',beta)

    if 1 <= beta && beta <= 2
        for j in 1:size(W,2)
            jcolW = view(W, : ,j)
            temp = @. W'[j, :] * (N[j, :]/(D[j, :]+eps()))
            jcolW .-= jcolW
            jcolW .+= (max.(epsilon, temp)) 
        end
    else
        if beta < 1
            gamma = 1/(2-beta)
        else
            gamma = 1/(beta-1) 
        end
        
        for j in 1:size(W,2)
            jcolW = view(W, : ,j)
            temp = @. W'[j, :] * ((N[j, :]/(D[j, :]+eps()))^gamma)
            jcolW .-= jcolW
            jcolW .+= (max.(epsilon, temp ))
        end
    end
end

# Update H by applying multiplicative update using N and D
# Multiplicative factor available on markdown file

function beta_updtH(X::AbstractMatrix{T},
                    W::AbstractMatrix{T},
                    Ht::AbstractMatrix{T};
                    beta::Float64=1.0,
                    epsilon::Float64=eps(),
                    args...
                    ) where T <: AbstractFloat

    N, D = num_denom(X, W, Ht', beta)

    if 1 <= beta && beta <= 2
        for i in 1:size(Ht,2)
            irowH = view(Ht, : ,i)
            temp = @. Ht[:, i] * (N[i, :]/(D[i, :]+eps()))
            irowH .-= irowH
            irowH .+= (max.(epsilon, temp)) 
        end
    else
        if beta < 1
            gamma = 1 / (2-beta)
        else
            gamma = 1 / (beta-1)
        end

        for i in 1:size(Ht,2)
            irowH = view(Ht, : ,i)
            temp = @. Ht[:, i] * (N[i, :]/(D[i, :] + eps()))^gamma
            irowH .-= irowH
            irowH .+= (max.(epsilon, temp))
        end
    end
end



# Subroutines

# Given the sparse matrix X, and the low-rank factors (W,H),
# Compute  Y = X .* (W*H).^beta
# This is the component-wise product between the sparse matrix X and the
# low-rank matrix WH component-wise exponentiated by beta.
# Since W*H can be dense, one should not compute W*H explicitely.
function compwiseprodsparselowrank(X::AbstractMatrix{T},
                                   W::AbstractMatrix{T},
                                   H::AbstractMatrix{T},
                                   fun::Function;
                                   args...) where T <: AbstractFloat

    l = findall(x -> x>0, A) #Indices of non-negative of A
    Y = sparse(size(X,1), size(X,2))
    for t in l
        Y[t] = fun(X[t] , (W[t[1],:]*H[:,t[2]]))
    end

    return Y
end

# Perform Y = compwiseprodsparselowrank(X,W,H,beta)
# by decomposing X into blocks
function blockrecursivecompwiseprodsparselowrank(X::AbstractMatrix{T},
                                                 W::AbstractMatrix{T},
                                                 H::AbstractMatrix{T},
                                                 fun::Function,
                                                 nnzparam::Float64 = 1e-3;
                                                 args...) where T <: AbstractFloat

    Y = sparse(size(X,1), size(X,2));

    if nnz(X) < nnzparam
        Y = compwiseprodsparselowrank(X,W,H,fun)

    else
        nblocks = 2
        m, n = size(X)
        mi = ceil(m / nblocks)
        ni = ceil(n / nblocks)
        for i = 1:nblocks
            for j = 1:nblocks
                indi = 1 + (i - 1) * mi : min(m, i*mi)
                indj = 1 + (j - 1) * ni : min(n, j*ni)
                Y[indi, indj] = blockrecursivecompwiseprodsparselowrank(X[indi, indj], W[indi,:], H[:,indj], fun, nnzparam )
            end
        end
    end

    return Y
end
