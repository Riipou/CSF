#Generation of matrix slackngon

function slackngon(n)
    line1 = cn(n, collect(0:(n-1)))
    S = zeros(n, n)
    shift = 1:n
    for i in 1:n
        S[i, shift] .= line1
        shift = [shift[2:end]; shift[1]]
    end
    r = nnrank(n)
    S = max.(S, 0)
    return S, r
end

function nnrank(n)
    k0 = ceil(log2(n))
    k1 = k0 - 1
    k2 = k0 - 2

    lb1 = 2^k1
    ub1 = 2^k1 + 2^k2
    lb2 = ub1
    ub2 = 2^k0

    if lb1 < n <= ub1
        return 2 * k0 - 1
    elseif lb2 < n <= ub2
        return 2 * k0
    end
end

function cn(n, k)
    return cos(pi/n) .- cos.(pi/n .+ 2 * pi * k/n)
end

function dn(n, k)
    return cn(n, k) / cn(n, 1)
end