import math
import numpy as np


def roots_third_degree(a, b, c, d):
    p = -(b ** 2 / (3 * a ** 2)) + c / a
    q = ((2 * b ** 3) / (27 * a ** 3)) - ((9 * c * b) / (27 * a ** 2)) + (d / a)
    delta = -(4 * p ** 3 + 27 * q ** 2)
    # DELTA < 0
    if delta < 0:
        u = (-q + math.sqrt(-delta / 27)) / 2
        v = (-q - math.sqrt(-delta / 27)) / 2
        if u < 0:
            u = -(-u) ** (1 / 3)
        elif u > 0:
            u = u ** (1 / 3)
        else:
            u = 0
        if v < 0:
            v = -(-v) ** (1 / 3)
        elif v > 0:
            v = v ** (1 / 3)
        else:
            v = 0
        root1 = u + v - (b / (3 * a))
        return [root1]
    # DELTA = 0
    elif delta == 0:
        if p == q == 0:
            root1 = 0
            return [root1]
        else:
            root1 = (3 * q) / p
            root2 = (-3 * q) / (2 * p)
            return [root1, root2]
    # DELTA > 0
    else:
        phi = math.acos(-q / 2 * math.sqrt(-27 / (p ** 3)))
        z1 = 2 * math.sqrt(-p / 3) * math.cos(phi / 3)
        z2 = 2 * math.sqrt(-p / 3) * math.cos((phi + 2 * math.pi) / 3)
        z3 = 2 * math.sqrt(-p / 3) * math.cos((phi + 4 * math.pi) / 3)
        root1 = z1 - (b / (3 * a))
        root2 = z2 - (b / (3 * a))
        root3 = z3 - (b / (3 * a))
        return [root1, root2, root3]


def init_matrix(M, r, choice):
    # Random U and V
    if choice == "random":
        U = np.random.rand(M.shape[0], r)
        V = np.random.rand(r, M.shape[1])
    # Calculation of U and V with SVD
    elif choice == "SVD":
        U, S, V = np.linalg.svd(M)
        U = U[:, :r]
        V = V[:r, :]
        U = np.dot(U, np.diag(S[:r]))
    return U, V


def calculate_function(a, b, c, d, x):
    return a * x ** 4 + b * x ** 3 + c * x ** 2 + d * x


def quartic_function(M, U, V, x_index, random_colonne, r):
    m, n = M.shape
    b = 0
    a = 0
    c = 0
    d = 0
    # Calculation of R
    reste = [0] * m
    for i in range(m):
        for index in range(r):
            if index != x_index:
                reste[i] += U[i][index] * V[index][random_colonne]
    # Calculation of coefficients for a fixed V variable
    for i in range(m):
        a += np.power(U[i][x_index], 4)
    for i in range(m):
        b_inter = 0
        b_inter += np.power(U[i][x_index], 3)
        b_inter = b_inter * reste[i]
        b += b_inter
    b *= 4
    for i in range(m):
        c_inter = 0
        c_inter += 3 * np.power(U[i][x_index], 2)
        c_inter_2 = reste[i] ** 2
        c_inter = c_inter * c_inter_2
        c_inter -= np.power(U[i][x_index], 2) * M[i][random_colonne]
        c += c_inter
    c *= 2
    for i in range(m):
        d_inter_1 = 0
        d_inter_2 = 0
        d_inter_1 += U[i][x_index]
        d_inter_2 += M[i][random_colonne] * U[i][x_index]
        d_inter_1 *= reste[i] ** 3
        d_inter_2 *= reste[i]
        d_inter_1 = d_inter_1 - d_inter_2
        d += d_inter_1
    d *= 4
    return a, b, c, d


def optimise_v(M, U, V):
    m, n = V.shape
    for i in range(n):
        for j in range(m):
            # We optimise the element [j,i]
            a, b, c, d = quartic_function(M, U, V, j, i, m)
            roots = roots_third_degree(4 * a, 3 * b, 2 * c, d)
            y = math.inf
            new_x = V[j, i]
            for root in roots:
                y_test = calculate_function(a, b, c, d, root)
                if y_test < y:
                    y = y_test
                    new_x = root
            if y < calculate_function(a, b, c, d, V[j, i]):
                V[j, i] = new_x
    return V


def coordinate_descent(max_iterations, M, U, V):
    alpha = 0.99
    for iteration in range(max_iterations):
        erreur_prec = np.linalg.norm(M - np.dot(U, V) ** 2)
        V = optimise_v(M, U, V)
        U = optimise_v(M.T, V.T, U.T).T
        erreur = np.linalg.norm(M - np.dot(U, V) ** 2)
        # stop the algorithm if no more changes
        if erreur > alpha * erreur_prec:
            break
    return U, V


def squared_factorisation(m, n, r, choice):
    # Number of iterations
    max_iterations = 10000
    # Number of tests to do
    nb_tests = 100
    nb_good = 0
    for i in range(nb_tests):
        # Creation of synthetic M
        U = np.random.rand(m, r)
        V = np.random.rand(r, n)
        M = np.dot(U, V) ** 2
        # Generation of matrix U and V
        U, V = init_matrix(M, r, choice)
        U, V = coordinate_descent(max_iterations, M, U, V)
        if (np.linalg.norm(M - np.dot(U, V) ** 2) / np.linalg.norm(M)) < 1e-3:
            nb_good += 1
    return nb_good / nb_tests


def multiple_test():
    # Choose between random and SVD
    choice1 = "random"
    choice2 = "SVD"
    # Rank
    r = 2
    # Sizes of the matrix m = n
    values = [5, 10, 50, 100]
    # Too long for more than 100
    with open("results/accuracy_multiple_tests_python.txt", "w") as file:
        file.write("RANDOM METHOD :\n")
        for i in values:
            print(i)
            m = n = i
            accuracy = squared_factorisation(m, n, r, choice1)
            file.write(f"m=n={n} accuracy={accuracy * 100}%\n")
            if accuracy == 0:
                break
        file.write("SVD METHOD :\n")
        for i in values:
            print(i)
            m = n = i
            accuracy = squared_factorisation(m, n, r, choice2)
            file.write(f"m=n={n} accuracy={accuracy * 100}%\n")
            if accuracy == 0:
                break


if __name__ == "__main__":
    multiple_test()
