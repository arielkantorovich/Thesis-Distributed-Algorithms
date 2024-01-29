import numpy as np
import os
from sklearn.preprocessing import PolynomialFeatures



def Initialize_action(L, N, K):
    """
    This function build action Matrix each player how much budget is spend on path Pi
    :param N: (int) number of players
    :param P: (int) number of paths from source to targer
    :return: Xe (2D-array) size NxP
    """
    Xn_k = np.random.rand(L, N, K).astype(np.float32)
    row_sums = np.sum(Xn_k, axis=2, keepdims=True)
    Xn_k /= row_sums
    return Xn_k

# # Define Path
# # file_path_Xn_k = os.path.join("Numpy_array_save", "N=50_K=5_congestion_game", "Pre_train_data", "Xn_k.npy")
file_path_Xe = os.path.join("Numpy_array_save", "N=50_K=5_congestion_game", "Pre_train_data", "Xe.npy")
file_path_Ck = os.path.join("Numpy_array_save", "N=50_K=5_congestion_game", "Pre_train_data", "Ck.npy")

file_path_Xe_0 = os.path.join("Numpy_array_save", "N=50_K=5_congestion_game", "Pre_train_data", "Xe_0.npy")
file_path_Ck_0 = os.path.join("Numpy_array_save", "N=50_K=5_congestion_game", "Pre_train_data", "Ck_0.npy")

file_path_Xe_1 = os.path.join("Numpy_array_save", "N=50_K=5_congestion_game", "Pre_train_data", "Xe_1.npy")
file_path_Ck_1 = os.path.join("Numpy_array_save", "N=50_K=5_congestion_game", "Pre_train_data", "Ck_1.npy")


file_path_C = os.path.join("Numpy_array_save", "N=50_K=5_congestion_game", "Pre_train_data", "C.npy")
file_path_B = os.path.join("Numpy_array_save", "N=50_K=5_congestion_game", "Pre_train_data", "B.npy")
file_path_A = os.path.join("Numpy_array_save", "N=50_K=5_congestion_game", "Pre_train_data", "A.npy")

# Read Data
Xe = np.load(file_path_Xe)
Ck = np.load(file_path_Ck)

Xe_0 = np.load(file_path_Xe_0)
Ck_0 = np.load(file_path_Ck_0)

Xe_1 = np.load(file_path_Xe_1)
Ck_1 = np.load(file_path_Ck_1)

A = np.load(file_path_A)
B = np.load(file_path_B)
C = np.load(file_path_C)

# L, K = Xe.shape
#
# X_Poly = np.concatenate([Xe, Xe_0, Xe_1], axis=-1)
# poly = PolynomialFeatures(2)
# X = poly.fit_transform(X_Poly)
X_train = np.concatenate([Ck, Xe, Xe ** 2,
                          Ck_0, Xe_0, Xe_0 ** 2,
                          Ck_1, Xe_1, Xe_1 ** 2], axis=-1)

Y_train = np.concatenate([A, B, C], axis=-1)

# Save results
np.save("Numpy_array_save/X_train.npy", X_train)
np.save("Numpy_array_save/Y_train.npy", Y_train)

