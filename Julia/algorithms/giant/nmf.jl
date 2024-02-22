export nmf

function nmf(X::AbstractMatrix{T},
             r::Integer;
             maxiter::Integer = 100,
             W0::AbstractMatrix{T} = zeros(T, 0, 0),
             H0::AbstractMatrix{T} = zeros(T, 0, 0),
             updaterW::Function = hals_updtW,
             updaterH::Function = hals_updtH,
             benchmark::Bool = false,
             objfunction::Function = fronorm,
             args...
             ) where T <: AbstractFloat
    # Constants
    m, n = size(X)
    # If not provided, init W and H randomly
    W0 = length(W0) == 0 ? rand(m, r) : W0
    W = copy(W0)
    H0 = length(H0) == 0 ? rand(r, n) : H0
    # Work on Ht to work along columns instead of rows (faster)
    Ht = copy(H0')

    # Main NMF loop
    times = zeros(Float64, maxiter)
    errors = zeros(Float64, maxiter)
    start_nmf = time()
    for it in 1:maxiter
        if benchmark
            times[it] = @elapsed begin
                updaterW(X, W, Ht; args...)
                updaterH(X, W, Ht; args...)
            end
            errors[it] = objfunction(X, W, Ht; args...)
            if it != 1
                times[it] += times[it-1]
            end
        else
            updaterW(X, W, Ht; args...)
            updaterH(X, W, Ht; args...)
        end
        if time()-start_nmf >= 60
            break
        end
    end
    if benchmark
        return W, Ht', times, errors
    else
        return W, Ht'
    end
end
