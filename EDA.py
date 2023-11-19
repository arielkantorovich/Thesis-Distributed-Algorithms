import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import PolynomialFeatures


# %% Read data and prepare to train
file_path_x = os.path.join("Numpy_array_save", "N=5_wireless", "X_train(bigData)_new.npy")
file_path_y = os.path.join("Numpy_array_save", "N=5_wireless", "Y_train.npy")
X_train = np.load(file_path_x)
Y_train = np.load(file_path_y)
N = 5
L = 30000

# Pat attention there is log connection
gn_n = X_train[:, 0]
In = X_train[:, 1]
phi_n = X_train[:, 2]
# Try build kernel connection K(Xi, Xj)
kernel = np.exp(- np.abs(In - gn_n) / 2)
# Build New X_train
X_new = np.zeros((N * L, 4))
X_new[:, 0] = gn_n
X_new[:, 1] = In
X_new[:, 2] = phi_n
X_new[:, 3] = kernel
# X_new[:, 4] = np.log(gn_n)
# X_new[:, 5] = np.log(In)
# X_new[:, 6] = np.sin(gn_n)
# X_new[:, 7] = np.cos(gn_n)
# X_new[:, 8] = np.tan(gn_n)
# X_new[:, 9] = np.sin(In)
# X_new[:, 10] = np.cos(In)
# X_new[:, 11] = np.tan(In)
# X_new[:, 12] = np.sin(phi_n)
# X_new[:, 13] = np.cos(phi_n)
# X_new[:, 14] = np.tan(phi_n)
# X_new[:, 15] = np.log(np.abs(In - gn_n))
np.save("Numpy_array_save/X_testNew(BigData).npy", X_new)
# plt.figure(1)
# plt.scatter(log_g_nn, Y_train), plt.xlabel('$log(g_(n,n))$'), plt.ylabel(r"$\beta_n$")
#
# plt.figure(2)
# plt.figure(figsize=(10, 6))
# plt.scatter(log_In, Y_train), plt.xlabel('$log(I_n)$'), plt.ylabel(r"$\beta_n$")
#
# plt.figure(3)
# plt.figure(figsize=(10, 6))
# plt.scatter(phi_n, Y_train), plt.xlabel(r'$\phi_n$'), plt.ylabel(r"$\beta_n $")
#
# plt.figure(4)
# plt.figure(figsize=(10, 6))
# plt.scatter(kernel, Y_train), plt.xlabel(r'$\ K(X_i, X_j))$'), plt.ylabel(r"$\beta_n $")
#
# plt.figure(5)
# plt.figure(figsize=(10, 6))
# plt.subplot(1, 2, 1)
# plt.scatter(np.sin(gn_n), Y_train), plt.xlabel(r'$\ sin(g_nn)$'), plt.ylabel(r"$\beta_n $")
# plt.subplot(1, 2, 2)
# plt.scatter(np.cos(gn_n), Y_train), plt.xlabel(r'$\ cos(g_nn)$'), plt.ylabel(r"$\beta_n $")
#
# plt.figure(6)
# plt.figure(figsize=(10, 6))
# plt.subplot(1, 2, 1)
# plt.scatter(np.sin(In)+np.cos(In), Y_train), plt.xlabel(r'$\ sin(In)$'), plt.ylabel(r"$\beta_n $")
# plt.subplot(1, 2, 2)
# plt.scatter(np.cos(In), Y_train), plt.xlabel(r'$\ cos(In)$'), plt.ylabel(r"$\beta_n $")
#
# plt.figure(7)
# plt.figure(figsize=(10, 6))
# plt.scatter(np.log(dist), Y_train), plt.xlabel(r'$|In-g_nn|$'), plt.ylabel(r"$\beta_n $")
#
# plt.figure(8)
# plt.figure(figsize=(10, 6))
# plt.scatter(np.tan(phi_n), Y_train), plt.xlabel(r'$sin(\phi_n)$'), plt.ylabel(r"$\beta_n $")
#
# plt.show()
print("Check")