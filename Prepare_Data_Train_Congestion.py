import numpy as np
import os

def EST_ABS(Ck, Xe, sigma=2):
    X_e_K = np.concatenate([np.ones_like(Xe).reshape(L, K, 1), Xe.reshape(L, K, 1), (Xe ** 2).reshape(L, K, 1)],axis=-1)
    SVD_est = np.linalg.pinv(X_e_K)
    ABC = np.matmul(SVD_est, Ck.reshape(L, K, 1))
    # Extract A, B, C Psudo inverse
    A = ABC[:, 0].squeeze(axis=-1)
    B = ABC[:, 1].squeeze(axis=-1)
    C = ABC[:, 2].squeeze(axis=-1)
    # # Expand such can add noise
    # A = np.repeat(A[:, np.newaxis], K, axis=-1) + np.random.randn(L, K) * sigma
    # B = np.repeat(B[:, np.newaxis], K, axis=-1) + np.random.randn(L, K) * sigma
    # C = np.repeat(C[:, np.newaxis], K, axis=-1) + np.random.randn(L, K) * sigma
    # # Expand such will coresspond to N players
    # A = np.repeat(A[:, np.newaxis, :], N, axis=1)
    # B = np.repeat(B[:, np.newaxis, :], N, axis=1)
    # C = np.repeat(C[:, np.newaxis, :], N, axis=1)
    return A, B, C

# Define Path
file_path_Xn_k = os.path.join("Numpy_array_save", "N=50_K=5_congestion_game", "Pre_data(L=500k)", "Xn_k.npy")
file_path_Xe = os.path.join("Numpy_array_save", "N=50_K=5_congestion_game", "Pre_data(L=500k)", "Xe.npy")
file_path_Ck = os.path.join("Numpy_array_save", "N=50_K=5_congestion_game", "Pre_data(L=500k)", "Ck.npy")
file_path_beta = os.path.join("Numpy_array_save", "N=50_K=5_congestion_game", "Pre_data(L=500k)", "beta.npy")
# Read Data
Xn_k = np.load(file_path_Xn_k)
Xe = np.load(file_path_Xe)
Ck = np.load(file_path_Ck)
beta = np.load(file_path_beta)

L, N, K = Xn_k.shape

# Extract A, B, C Additional information
# A, B, C = EST_ABS(Ck[:, 0, :], Xe[:, 0, :], sigma=2)
# A = np.repeat(A, N)
# B = np.repeat(B, N)
# C = np.repeat(C, N)



# For CNN
# Reshape each player's data into matrices of size kx3
# xn_k_reshaped = Xn_k.reshape(L * N, K, 1)
# xe_reshaped = Xe.reshape(L * N, K, 1)
# ck_reshaped = Ck.reshape(L * N, K, 1)
# reshaped_data = np.concatenate((xn_k_reshaped, xe_reshaped, ck_reshaped), axis=2)


# # Concatenate along the last axis
ONE = np.ones_like(Xe)
X_train = np.concatenate([Xn_k, Ck, ONE, Xe, Xe ** 2], axis=-1)
X_train_reshaped = np.reshape(X_train, (L * N, 5 * K))
# X_train = np.zeros((L * N, 5 * K + 3))
# X_train[:, 0:(5*K)] = X_train_reshaped
# X_train[:, -1] = C
# X_train[:, -2] = B
# X_train[:, -3] = A
# Prepare Y train
# Y = np.reshape(beta, (L * N, K))
# Save Data for training
np.save("Numpy_array_save/X_train_new(L=500k).npy", X_train_reshaped)
# np.save("Numpy_array_save/Y_train(L=500k).npy", Y)

print("Finsh ! ! !")

