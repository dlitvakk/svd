import numpy as np


def gram_schmidt(vectors):
    # number of vectors
    num_vectors = vectors.shape[0]

    # array for all the vectors
    ortho_vectors = np.zeros_like(vectors,
                                  dtype=float)

    for i in range(num_vectors):
        # starting with the vector we have in beginning
        ortho_vector = vectors[i]

        # by formula evaluating other vectors
        for j in range(i):
            proj = np.dot(ortho_vectors[j],
                          vectors[i]) / np.dot(ortho_vectors[j],
                                               ortho_vectors[j])
            ortho_vector -= proj * ortho_vectors[j]

        # updating the vector
        ortho_vectors[i] = ortho_vector

    # normalizing vectors (by dividing with their norm)
    for i in range(num_vectors):
        norm = np.linalg.norm(ortho_vectors[i])
        if norm != 0:
            ortho_vectors[i] /= norm

    return ortho_vectors


def calculate_svd(matrix):
    # A (m*n) -> calculating A(T)A (n*n) and AA(T) (m*m)
    ata = matrix.T @ matrix
    aat = matrix @ matrix.T

    # calculating eigenvalues and eigenvectors
    ata_eigenvalues, ata_eigenvectors = np.linalg.eig(ata)
    aat_eigenvalues, aat_eigenvectors = np.linalg.eig(aat)

    # sorting those values
    sorted_indices_ata = np.argsort(ata_eigenvalues)[::-1]
    sorted_indices_aat = np.argsort(aat_eigenvalues)[::-1]

    sorted_ata_eigenvalues = ata_eigenvalues[sorted_indices_ata]
    sorted_aat_eigenvalues = aat_eigenvalues[sorted_indices_aat]

    sorted_ata_eigenvectors = ata_eigenvectors[:, sorted_indices_ata]
    sorted_aat_eigenvectors = aat_eigenvectors[:, sorted_indices_aat]

    # Σ is square root of A(T)A eigenvalues in descending order
    e = np.zeros(matrix.shape)
    min_dim = min(matrix.shape)
    e[:min_dim, :min_dim] = np.diag(np.sqrt(sorted_ata_eigenvalues[:min_dim]))

    # U is eigenvectors of AAT
    u = sorted_aat_eigenvectors

    # V is eigenvectors of ATA
    v = sorted_ata_eigenvectors

    # making U and V  orthonormal
    u = gram_schmidt(u.T).T
    v = gram_schmidt(v.T).T

    # we need V(T)
    vt = v.T

    # Ensure that σ * u_i = A * v_i for singular vectors
    for i in range(min_dim):
        ui = matrix @ v[:, i]
        norm_ui = np.linalg.norm(ui)
        if norm_ui != 0:
            u[:, i] = ui / norm_ui
            e[i, i] = norm_ui

    return u, e, vt


A = np.array([[1, 2], [3, 4], [5, 6]])
U, Σ, VT = calculate_svd(A)

print("U:\n",
      U)
print("Σ:\n",
      Σ)
print("VT:\n",
      VT)

# checking if working fine
print("The original matrix:\n", A)
A_2 = U @ Σ @ VT
print("A(SVD):\n",
      A_2)
