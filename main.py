import numpy as np


def roots_third_degree(a, b, c, d):
    # Définition des coefficients
    coefficients = [a, b, c, d]
    # Trouver les racines du polynomes
    roots = np.real(np.roots(coefficients)[np.isreal(np.roots(coefficients))])
    return roots


def init_matrix():
    A = np.array([[4, 5, 2, 6],
                  [9, 4, 5, 6],
                  [4, 3, 6, 9],
                  [10, 1, 6, 4],
                  [1, 7, 9, 11]])
    r = 2
    U = np.random.rand(A.shape[0], r)
    V = np.random.rand(r, A.shape[1])
    return A, U, V


def main():
    roots = roots_third_degree(4, -3, 0, 1)
    print("Les racines du polynomes sont:", roots)
    A, U, V = init_matrix()
    print("Matrice A :")
    for row in A:
        print(row)
    print("Matrice U initiale :")
    for row in U:
        print(row)
    print("Matrice V initiale :")
    for row in V:
        print(row)


if __name__ == "__main__":
    main()
