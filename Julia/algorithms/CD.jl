using LinearAlgebra
include("functions.jl")
function fourth_degree_polynomial(a, b, c, d, x)
    return a*x^4+b*x^3+c*x^2+d*x
end

function roots_third_degree(a, b, c, d)
    
    if a == 0
        if b == 0
            root1 = -d/c
            return root1
        end
        delta = c^2-4*b*d
        root1 = (-c + sqrt(delta))/ (2 * b)
        root2 = (-c - sqrt(delta))/ (2 * b)
        f_root1 = fourth_degree_polynomial(a/4, b/3, c/2, d, root1)
        f_root2 = fourth_degree_polynomial(a/4, b/3, c/2, d, root2)
        if argmin([f_root1, f_root2]) == 1
            return root1
        else
            return root2
        end
    end

    p = -(b^2 / (3 * a^2)) + c / a
    q = ((2 * b^3) / (27 * a^3)) - ((9 * c * b) / (27 * a^2)) + (d / a)
    delta = -(4 * p^3 + 27 * q^2)
    if delta < 0
        u = (-q + sqrt(-delta / 27)) / 2
        v = (-q - sqrt(-delta / 27)) / 2
        if u < 0
            u = -(-u)^(1 / 3)
        elseif u > 0
            u = u^(1 / 3)
        else
            u = 0
        end
        if v < 0
            v = -(-v)^(1 / 3)
        elseif v > 0
            v = v^(1 / 3)
        else
            v = 0
        end
        root1 = u + v - (b / (3 * a))
        return root1
    elseif delta == 0
        if p == q == 0
            root1 = 0
            return root1
        else
            root1 = (3 * q) / p
            root2 = (-3 * q) / (2 * p)
            f_root1 = fourth_degree_polynomial(a/4, b/3, c/2, d, root1)
            f_root2 = fourth_degree_polynomial(a/4, b/3, c/2, d, root2)
            if argmin([f_root1, f_root2]) == 1
                return root1
            else
                return root2
            end
        end
    else
        epsilon = -1e-300
        phi = acos(-q / 2 * sqrt(-27 / (p^3 + epsilon)))
        z1 = 2 * sqrt(-p / 3) * cos(phi / 3)
        z2 = 2 * sqrt(-p / 3) * cos((phi + 2 * π) / 3)
        z3 = 2 * sqrt(-p / 3) * cos((phi + 4 * π) / 3)
        root1 = z1 - (b / (3 * a))
        root2 = z2 - (b / (3 * a))
        root3 = z3 - (b / (3 * a))
        f_root1 = fourth_degree_polynomial(a/4, b/3, c/2, d, root1)
        f_root2 = fourth_degree_polynomial(a/4, b/3, c/2, d, root2)
        f_root3 = fourth_degree_polynomial(a/4, b/3, c/2, d, root3)
        if argmin([f_root1, f_root2, f_root3]) == 1
            return root1
        elseif argmin([f_root1, f_root2, f_root3]) == 2
            return root2
        else
            return root3
        end
    end
end

function cs_least_square(M, U, V, r, j, V_t)
    R = U*V[:,j]
    for p in 1:r
        if V_t[p,j] == 0
            R -= U[:,p]*V[p,j]
            m, _ = size(M)
            c3, c2, c1, c0 = 0, 0, 0, 0
            for i in 1:m
                c3 += U[i, p]^4
                c2 += U[i, p]^3 * R[i]
                c1 += 3 * U[i, p]^2 * R[i]^2 - U[i, p]^2 * M[i, j]
                c0 += U[i, p] * R[i]^3 - M[i, j] * U[i, p] * R[i]
            end
            c2 *= 4
            c1 *= 2
            c0 *= 4
            V[p, j] = roots_third_degree(4 * c3, 3 * c2, 2 * c1, c0)
            R += U[:,p]*V[p,j]
        end
    end
    return V
end

function optimise_v(M, U, V, V_t)
    r, n = size(V)
    for j in 1:n
        V = cs_least_square(M, U, V, r, j, V_t)
    end
    return V
end

function coordinate_descent(
    max_iterations::Int,
    M::Matrix,
    U::Matrix,
    V::Matrix;
    max_time:: Int = 0,
    alpha:: Float64 = Inf,
    indices_V:: Vector{Tuple{Int, Int}} = Vector{Tuple{Int, Int}}(),
    indices_U:: Vector{Tuple{Int, Int}} = Vector{Tuple{Int, Int}}(),
    errors_calculation::Bool = false)


    r, n = size(V)
    V_t = zeros(r, n)

    m, r = size(U)
    U_t = zeros(m, r)

    if errors_calculation
        errors = []
    end

    if !isempty(indices_V)
        for idx_v in indices_V
            row_idx, col_idx = idx_v
            V_t[row_idx, col_idx] = 1
        end
    end

    if !isempty(indices_U)
        for idx_u in indices_U
            row_idx, col_idx = idx_u
            U_t[row_idx, col_idx] = 1
        end
    end
    
    if max_time == 0
        max_time = Inf
    end
    
    if alpha <= 1
        prev_error = norm(M - (U * V).^2)
    end
   
    start = time()

    for ite in 1:max_iterations

        V = optimise_v(M, U, V, V_t)
        U = optimise_v(M', V', U', U_t')'

        if alpha <=1
            error = norm(M - (U * V).^2)
        end
        
        if errors_calculation
            push!(errors, norm(M - (U * V).^2)/norm(M))
        end
        
        if ite % 10 == 0 && alpha <= 1
            if error > alpha * prev_error
                break
            end
            prev_error = norm(M - (U * V).^2)
        end

        if time()-start >= max_time
            break
        end
        
    end
    
    if errors_calculation
        return errors
    else
        return U, V
    end
end

function coordinate_descent_extrapoled(
    max_iterations::Int,
    M::Matrix,
    U::Matrix,
    V::Matrix;
    max_time:: Int = 0,
    alpha:: Float64 = Inf,
    indices_V:: Vector{Tuple{Int, Int}} = Vector{Tuple{Int, Int}}(),
    indices_U:: Vector{Tuple{Int, Int}} = Vector{Tuple{Int, Int}}(),
    beta_bis::Float64 = 0.3,
    eta::Float64 = 1.5,
    gamma::Float64 = 1.05,
    gamma_bis::Float64 = 1.01,
    errors_calculation:: Bool = false)

    r, n = size(V)
    V_t = zeros(r, n)

    m, r = size(U)
    U_t = zeros(m, r)

    if errors_calculation
        errors = []
        times = []
    end

    if !isempty(indices_V)
        for idx_v in indices_V
            row_idx, col_idx = idx_v
            V_t[row_idx, col_idx] = 1
        end
    end

    if !isempty(indices_U)
        for idx_u in indices_U
            row_idx, col_idx = idx_u
            U_t[row_idx, col_idx] = 1
        end
    end
    
    if max_time == 0
        max_time = Inf
    end
    
    prev_error = norm(M - (U * V).^2)

    start = time()

    Up = copy(U)
    Vp = copy(V)
    Vtemp = copy(V)
    Utemp = copy(U)
    beta = 1
    beta_pred = beta_bis

    for ite in 1:max_iterations
        V = optimise_v(M, U, V + beta_bis * (V - Vp), V_t)
        Vp = copy(Vtemp)
        U = optimise_v(M', V', U' + beta_bis * (U' - Up'), U_t')'
        Up = copy(Utemp)
        Utemp = copy(U)
        Vtemp = copy(V)

        error = norm(M - (U * V).^2)
        

        if error > norm(M - (Up*Vp).^2)
            V = optimise_v(M, Up, Vp, V_t)
            U = optimise_v(M', V', Up', U_t')'
            Utemp = copy(U)
            Vtemp = copy(V)
            error = norm(M - (U*V).^2)
            beta_copy = copy(beta_bis)
            beta_bis = beta_bis/eta
            beta = beta_pred
            beta_pred = beta_copy
        else
            beta_pred = beta_bis
            beta_bis = min(beta, gamma * beta_bis)
            beta = min(1, gamma_bis * beta)
        end

        if errors_calculation
            push!(errors, error/norm(M))
            push!(times, time()-start)
        end

        if ite % 10 == 0 && alpha <= 1
            if error > alpha * prev_error
                break
            end
            prev_error = norm(M - (U * V).^2)
        end
        
        if time()-start >= max_time
            break
        end           
    end

    if errors_calculation
        return errors, times
    else
        return U, V
    end
end