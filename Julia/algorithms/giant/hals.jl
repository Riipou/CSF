# Hierarchical Alternating Least Squares

export hals_updtW, hals_updtH


function hals_updtW(X::AbstractMatrix{T},
                    W::AbstractMatrix{T},
                    Ht::AbstractMatrix{T},
                    XHt::AbstractMatrix{T}=zeros(T, 0, 0),
                    HHt::AbstractMatrix{T}=zeros(T, 0, 0);
                    args...
                    ) where T <: AbstractFloat
    # Init
    r = size(W, 2)
    # If needed, compute intermediary values
    if length(XHt) == 0
        XHt = X * Ht
    end
    if length(HHt) == 0
        HHt = Ht' * Ht
    end

    # Loop on columns of W
    for j in 1:r
        jcolW = view(W, :, j)
        deltaW = max.((XHt[:,j] - W * HHt[:,j]) / HHt[j,j], -jcolW)
        jcolW .+= deltaW
        # Safety procedure to avoid div by 0
        if iszero(jcolW)
            jcolW .= 1e-15
        end
    end
end


function hals_updtH(X::AbstractMatrix{T},
                    W::AbstractMatrix{T},
                    Ht::AbstractMatrix{T},
                    XtW::AbstractMatrix{T}=zeros(T, 0, 0),
                    WtW::AbstractMatrix{T}=zeros(T, 0, 0);
                    args...
                    ) where T <: AbstractFloat
    # Init
    r = size(Ht, 2)
    # If needed, compute intermediary values
    if length(XtW) == 0
        XtW = X' * W
    end
    if length(WtW) == 0
        WtW = W' * W
    end

    # Loop on rows of H (columns of Ht)
    for i in 1:r
        irowH = view(Ht, :, i)
        deltaH = max.((XtW[:,i] - Ht * WtW[:,i]) / WtW[i,i], -irowH)
        irowH .+= deltaH
        # Safety procedure to avoid div by 0
        if iszero(irowH)
            irowH .= 1e-15
        end
    end
end
