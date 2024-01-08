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

function quartic_function(M, U, V, x_index, random_colonne, r)
    m, n = size(M)
    b = 0
    a = 0
    c = 0
    d = 0
    
    reste = zeros(m)
    for i in 1:m
        for index in 1:r
            if index != x_index
                reste[i] += U[i, index] * V[index, random_colonne]
            end
        end
    end
    
    for i in 1:m
        a += U[i, x_index]^4
    end
    
    for i in 1:m
        b_inter = 0
        b_inter += U[i, x_index]^3
        b_inter = b_inter * reste[i]
        b += b_inter
    end
    b *= 4
    
    for i in 1:m
        c_inter = 0
        c_inter += 3 * U[i, x_index]^2
        c_inter_2 = reste[i]^2
        c_inter = c_inter * c_inter_2
        c_inter -= U[i, x_index]^2 * M[i, random_colonne]
        c += c_inter
    end
    c *= 2
    
    for i in 1:m
        d_inter_1 = 0
        d_inter_2 = 0
        d_inter_1 += U[i, x_index]
        d_inter_2 += M[i, random_colonne] * U[i, x_index]
        d_inter_1 *= reste[i]^3
        d_inter_2 *= reste[i]
        d_inter_1 = d_inter_1 - d_inter_2
        d += d_inter_1
    end
    d *= 4
    return a, b, c, d
end

function optimise_v(M, U, V)
    m, n = size(V)
    for i in 1:n
        for j in 1:m
            a, b, c, d = quartic_function(M, U, V, j, i, m)
            roots = roots_third_degree(4 * a, 3 * b, 2 * c, d)
            y = Inf
            new_x = V[j, i]
            for root in roots
                y_test = calculate_function(a, b, c, d, root)
                if y_test < y
                    y = y_test
                    new_x = root
                end
            end
            if y < calculate_function(a, b, c, d, V[j, i])
                V[j, i] = new_x
            end
        end
    end
    return V
end

function coordinate_descent(max_iterations, M, U, V)
    alpha = 0.99
    for iteration in 1:max_iterations
        erreur_prec = norm(M - (U * V).^2)
        V = optimise_v(M, U, V)
        U = optimise_v(M', V', U')'
        erreur = norm(M - (U * V).^2)
        if erreur > alpha * erreur_prec
            break
        end
    end
    return U, V
end