import numpy as np


def roots_third_degree(a, b, c, d):
    # Définition des coefficients
    coefficients = [a, b, c, d]
    # Trouver les racines du polynomes
    roots = np.real(np.roots(coefficients)[np.isreal(np.roots(coefficients))])
    return roots


def init_matrix():
    M = np.array([[4, 5, 2, 6],
                  [9, 4, 5, 6],
                  [4, 3, 6, 9],
                  [10, 1, 6, 4],
                  [1, 7, 9, 11]])
    r = 2
    U = np.random.rand(M.shape[0], r)
    V = np.random.rand(r, M.shape[1])
    return M, U, V


def quartic_function(M, U, V):
    # On calcule pour la variable x1
    m, n = M.shape
    a = 0
    b = 0
    for i in range(m):
        a += np.power(U[i][0], 4)
    for i in range(m):
        b += np.power(U[i][0], 3)*U[i][1]*V[1][0]


def main():
    M, U, V = init_matrix()
    print("Matrice M :")
    for row in M:
        print(row)
    print("Matrice U initiale :")
    for row in U:
        print(row)
    print("Matrice V initiale :")
    for row in V:
        print(row)
    quartic_function(M, U, V)


if __name__ == "__main__":
    main()
