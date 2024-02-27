using LinearAlgebra 

function roots_third_degree(a, b, c, d)
    
    if a == 0
        if b == 0
            root1 = -d/c
            return [root1]
        end
        delta = c^2-4*b*d
        root1 = (-c + sqrt(delta))/ (2 * b)
        root2 = (-c - sqrt(delta))/ (2 * b)
        if root1 == root2
            return [root1]
        else
            return [root1, root2]
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
        return [root1]
    elseif delta == 0
        if p == q == 0
            root1 = 0
            return [root1]
        else
            root1 = (3 * q) / p
            root2 = (-3 * q) / (2 * p)
            return [root1, root2]
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
        return [root1, root2, root3]
    end
end

function init_matrix(M, r, choice)
    if choice == "random"
        U = rand(size(M, 1), r)
        V = rand(r, size(M, 2))
        A = (U*V).^2
        lambda = sum(A[:].*M[:])/sum(A[:].^2)
        U *= lambda
        V *= lambda
    elseif choice == "SVD"
        U, S, V = svd(M)
        V = V'
        U = U[:, 1:r]
        V = V[1:r, :]
        U = U * Diagonal(sqrt.(S[1:r]))
        V = Diagonal(sqrt.(S[1:r])) * V
    end
    return U, V
end

function cs_least_square(M, U, V, r, j)
    R = U*V[:,j]
    for p in 1:r
        R -= U[:,p]*V[p,j]
        m, _ = size(M)
        a, b, c, d = 0, 0, 0, 0
        for i in 1:m
            a += U[i, p]^4
            b += U[i, p]^3 * R[i]
            c += 3 * U[i, p]^2 * R[i]^2 - U[i, p]^2 * M[i, j]
            d += U[i, p] * R[i]^3 - M[i, j] * U[i, p] * R[i]
        end
        b *= 4
        c *= 2
        d *= 4
        roots = roots_third_degree(4 * a, 3 * b, 2 * c, d)
        y = Inf
        new_x = V[p, j]
        for root in roots
            y_test = a * root^4 + b * root^3 + c * root^2 + d * root
            if y_test < y
                y = y_test
                new_x = root
            end
        end
        V[p, j] = new_x
        R += U[:,p]*V[p,j]
    end
    return V
end

function optimise_v(M, U, V)
    r, n = size(V)
    for j in 1:n
        V = cs_least_square(M, U, V, r, j)
    end
    return V
end

function coordinate_descent(
    max_iterations::Int,
    M::Matrix,
    U::Matrix,
    V::Matrix;
    max_time:: Int = 0,
    alpha:: Float64 = Inf)

    if max_time == 0
        max_time = Inf
    end
    prev_error = norm(M - (U * V).^2)
    start = time()
    for ite in 1:max_iterations
        
        V = optimise_v(M, U, V)
        U = optimise_v(M', V', U')'

        error = norm(M - (U * V).^2)
        if ite % 10 == 0
            if error > alpha * prev_error
                break
            end
            prev_error = norm(M - (U * V).^2)
        end
       
        if time()-start >= max_time
            break
        end
        
    end
    return U, V
end