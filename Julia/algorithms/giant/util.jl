# Some general utility functions

using Random, LinearAlgebra

export logrange, fronorm, rand_ksparse, rand_ksparse_vect, xdy, betadiv

# Function similar to Matlab's logspace
function logrange(x1, x2, n)
    collect(10^y for y in range(log10(x1), log10(x2), length=n))
end

# The Frobenius norm of X-WH
function fronorm(X, W, Ht)
    return sum(X.*X) - 2*sum(W.*(X*Ht)) + sum((W'*W).*(Ht'*Ht))
end

"""
    rand_ksparse(m, n, k)

Generates a random matrix of size m*n with exactly k nonzero entries by column.
"""
function rand_ksparse(m, n, k)
    mat = zeros(m, n)
    for col in eachcol(mat)
        col .= rand_ksparse_vect(m, k)
    end
    return mat
end


"""
    rand_ksparse_vect(n, k)

Generates a random vector of size n with exactly k nonzero entries.
"""
function rand_ksparse_vect(n, k)
    k <= n || error("k is greater than n")
    vect = zeros(n)
    perm = randperm(n)[1:k]
    for i in perm
        vect[i] = rand()
    end
    return vect
end

function xdy(x,y)
    return x / (y .+ eps())
end

# Computation of the beta-divergence between X and Y, that is, 
# dist = D_beta(X,Y) = sum(Z(:)) where Z is the component-wise divergence 
# between the entries of X and Y.
function betadiv(X, W, Ht; beta)
    Y = W * Ht'
    if beta == 0          # Itakura–Saito distance 
        XsY = X ./ (Y .+ eps())
        Z = XsY .- log.(XsY .+ eps()) .- 1
        dist = sum(Z)
    elseif beta == 1      # Kullback-Leibler divergence
        Z = X .* log.(X ./ (Y .+ eps()) .+ eps()) .- X .+ Y
        dist = sum(Z)
    else                  # Other beta divergences
        Z = ((max.(X, eps())).^beta + (beta - 1) * (Y.^beta) - beta * X .* (Y.^(beta-1))) / (beta * (beta - 1))
        dist = sum(Z)
    end

    return dist
end
