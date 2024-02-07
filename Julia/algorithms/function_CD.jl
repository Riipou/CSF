using LinearAlgebra

function roots_third_degree(a, b, c, d)
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
        phi = acos(-q / 2 * sqrt(-27 / (p^3)))
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
    elseif choice == "SVD"
        U, S, V = svd(M)
        U = U[:, 1:r]
        V = V[1:r, :]
        U = U * Diagonal(S[1:r])
    end
    return U, V
end

function calculate_function(a, b, c, d, x)
    return a * x^4 + b * x^3 + c * x^2 + d * x
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
            y_test = calculate_function(a, b, c, d, root)
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

function coordinate_descent(max_iterations, M, U, V, alpha)
    for _ in 1:max_iterations
        prev_error = norm(M - (U * V).^2)
        V = optimise_v(M, U, V)
        U = optimise_v(M', V', U')'
        error = norm(M - (U * V).^2)
        if error > alpha * prev_error
            break
        end
    end
    return U, V
end