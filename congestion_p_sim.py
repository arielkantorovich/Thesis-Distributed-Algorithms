import matplotlib.pyplot as plt
import numpy as np

def Generate_Game_Param(L, K):
    """
    :param K:(int), Number of strategy in game
    :return:
    """
    A = np.random.uniform(low=0.0, high=1.0, size=(L, K))
    B = np.random.uniform(low=0.0, high=1.0, size=(L, K))
    C = np.random.uniform(low=0.0, high=1.0, size=(L, K))
    return A, B, C

def Initialize_action(L, N, K):
    """
    This function build action Matrix each player how much budget is spend on path Pi
    :param N: (int) number of players
    :param P: (int) number of paths from source to targer
    :return: Xe (2D-array) size NxP
    """
    Xn_k = np.random.uniform(low=0.0, high=1.0, size=(L, N, K))
    row_sums = np.sum(Xn_k, axis=2, keepdims=True)
    Xn_k /= row_sums
    return Xn_k

def calc_gradient(C_k, Xn_k, B, C, Xe, Nash_flag=False, beta=0):
    """
    :param C_k: size (L, K, )
    :param Xn_k: size (L, N, K)
    :param B: size (L, K,)
    :param C: size (L, K,)
    :param Nash_flag: bool, specify if calculate only gradient Nash
    :param beta: size (L, K,)
    :return: gradient of each player size (N, )
    """
    N = Xn_k.shape[1]
    if Nash_flag:
        self_gradient = C_k * (Xn_k > 0) + Xn_k * np.repeat((B + 2 * C * Xe)[:, np.newaxis, :], N, axis=1) + beta * (Xn_k > 0)
        return self_gradient
    sum_Xn_k_expanded = np.repeat(np.sum(Xn_k, axis=1)[:, np.newaxis, :], N, axis=1)
    global_grad = C_k * (Xn_k > 0) + np.repeat((B + 2 * C * Xe)[:, np.newaxis, :], N, axis=1) * sum_Xn_k_expanded * (Xn_k > 0)
    return global_grad


def project_onto_simplex(V, z=1):
    """
    Project matrix X onto the probability simplex.
    :param V: numpy array, matrix of size (LxNxK)
    :param z: integer, norm of projected vector
    :return X: numpy array, matrix of the same size as X, projected onto the probability simplex
    """
    L, N, K = V.shape
    U = np.sort(V, axis=-1)[:, :, ::-1]
    z_vector = np.ones((L, N)) * z
    cssv = np.cumsum(U, axis=-1) - z_vector[:, :, np.newaxis]
    ind = np.arange(K) + 1
    cond = U - cssv / ind > 0
    rho = np.count_nonzero(cond, axis=-1)
    theta = cssv[np.arange(L)[:, np.newaxis], np.arange(N), rho - 1] / rho
    result = np.maximum(V - theta[:, :, np.newaxis], 0)
    return result

def compute_grad_p(xk, Ck, bk, dk, p=0, N=20):
    """
    This function calculate the grad tax from Ck pow p
    :param xk: (L, K)
    :param Ck: (L, K)
    :param bk: (L, K)
    :param dk: (L, K)
    :param p: (int)
    :param N: (int)
    :return: grad (L, K)
    """
    grad = p * (Ck ** (p - 1)) * (bk + 2 * dk * xk)
    grad = np.repeat(grad[:, np.newaxis, :], N, axis=1)
    return grad


# Hyper-Parameters
N = 20 # Number of players
T = 500
L = 2000
K = 5 # Number of strategy
lr = 0.03 * np.reciprocal(np.power(range(1, T + 1), 0.75))
P_list = [0.2, 0.3, 0.5, 2, 3, 5]

# Record Final Cost
global_record = np.zeros((T, L), dtype=np.float64)
nash_1_record = np.zeros((T, L), dtype=np.float32)
nash_2_record = np.zeros((T, L), dtype=np.float32)
nash_3_record = np.zeros((T, L), dtype=np.float32)
nash_4_record = np.zeros((T, L), dtype=np.float32)
nash_5_record = np.zeros((T, L), dtype=np.float32)
nash_6_record = np.zeros((T, L), dtype=np.float32)
nash_record = np.zeros((T, L), dtype=np.float32)

# Generate A, B, C Parameters
A, B, C = Generate_Game_Param(L, K)

# Generate action of players
global_Xnk = Initialize_action(L, N, K)
nash_1_Xnk = global_Xnk.copy()
nash_2_Xnk = global_Xnk.copy()
nash_3_Xnk = global_Xnk.copy()
nash_4_Xnk = global_Xnk.copy()
nash_5_Xnk = global_Xnk.copy()
nash_6_Xnk = global_Xnk.copy()
nash_Xnk = global_Xnk.copy()

# Main loop
for t in range(T):
    # Calculate Xk
    global_Xk = np.sum(global_Xnk, axis=1)
    nash_Xk = np.sum(nash_Xnk, axis=1)
    nash_1_Xk = np.sum(nash_1_Xnk, axis=1)
    nash_2_Xk = np.sum(nash_2_Xnk, axis=1)
    nash_3_Xk = np.sum(nash_3_Xnk, axis=1)
    nash_4_Xk = np.sum(nash_4_Xnk, axis=1)
    nash_5_Xk = np.sum(nash_5_Xnk, axis=1)
    nash_6_Xk = np.sum(nash_6_Xnk, axis=1)
    # calculate congestion
    Ck_global = A + B * global_Xk + C * (global_Xk ** 2)
    Ck_nash = A + B * nash_Xk + C * (nash_Xk ** 2)
    Ck_nash_1 = A + B * nash_1_Xk + C * (nash_1_Xk ** 2)
    Ck_nash_2 = A + B * nash_2_Xk + C * (nash_2_Xk ** 2)
    Ck_nash_3 = A + B * nash_3_Xk + C * (nash_3_Xk ** 2)
    Ck_nash_4 = A + B * nash_4_Xk + C * (nash_4_Xk ** 2)
    Ck_nash_5 = A + B * nash_5_Xk + C * (nash_5_Xk ** 2)
    Ck_nash_6 = A + B * nash_6_Xk + C * (nash_6_Xk ** 2)
    # Expand C_k such is corresponded to handle with L games parallel
    Ck_global_expand = np.repeat(Ck_global[:, np.newaxis, :], N, axis=1)
    Ck_nash_expand = np.repeat(Ck_nash[:, np.newaxis, :], N, axis=1)
    Ck_nash_1_expand = np.repeat(Ck_nash_1[:, np.newaxis, :], N, axis=1)
    Ck_nash_2_expand = np.repeat(Ck_nash_2[:, np.newaxis, :], N, axis=1)
    Ck_nash_3_expand = np.repeat(Ck_nash_3[:, np.newaxis, :], N, axis=1)
    Ck_nash_4_expand = np.repeat(Ck_nash_4[:, np.newaxis, :], N, axis=1)
    Ck_nash_5_expand = np.repeat(Ck_nash_5[:, np.newaxis, :], N, axis=1)
    Ck_nash_6_expand = np.repeat(Ck_nash_6[:, np.newaxis, :], N, axis=1)
    # calculate cost of player n
    Cn_global = np.sum(global_Xnk * Ck_global_expand, axis=2)
    Cn_nash = np.sum(nash_Xnk * Ck_nash_expand, axis=2)
    Cn_nash_1 = np.sum(nash_1_Xnk * Ck_nash_1_expand, axis=2)
    Cn_nash_2 = np.sum(nash_2_Xnk * Ck_nash_2_expand, axis=2)
    Cn_nash_3 = np.sum(nash_3_Xnk * Ck_nash_3_expand, axis=2)
    Cn_nash_4 = np.sum(nash_4_Xnk * Ck_nash_4_expand, axis=2)
    Cn_nash_5 = np.sum(nash_5_Xnk * Ck_nash_5_expand, axis=2)
    Cn_nash_6 = np.sum(nash_6_Xnk * Ck_nash_6_expand, axis=2)
    # Calculate Global & Nash record
    global_record[t] = np.sum(Cn_global, axis=1)
    nash_record[t] = np.sum(Cn_nash, axis=1)
    nash_1_record[t] = np.sum(Cn_nash_1, axis=1)
    nash_2_record[t] = np.sum(Cn_nash_2, axis=1)
    nash_3_record[t] = np.sum(Cn_nash_3, axis=1)
    nash_4_record[t] = np.sum(Cn_nash_4, axis=1)
    nash_5_record[t] = np.sum(Cn_nash_5, axis=1)
    nash_6_record[t] = np.sum(Cn_nash_6, axis=1)
    # calculate gradient
    grad_global = calc_gradient(Ck_global_expand, global_Xnk, B, C, global_Xk, Nash_flag=False)
    grad_nash = calc_gradient(Ck_nash_expand, nash_Xnk, B, C, nash_Xk, Nash_flag=True, beta=0)

    beta_1 = compute_grad_p(nash_1_Xk, Ck_nash_1, B, C, p=0.2, N=20)
    grad_nash_1 = calc_gradient(Ck_nash_1_expand, nash_1_Xnk, B, C, nash_1_Xk, Nash_flag=True, beta=beta_1)

    beta_2 = compute_grad_p(nash_2_Xk, Ck_nash_2, B, C, p=0.3, N=20)
    grad_nash_2 = calc_gradient(Ck_nash_2_expand, nash_2_Xnk, B, C, nash_2_Xk, Nash_flag=True, beta=beta_2)

    beta_3 = compute_grad_p(nash_3_Xk, Ck_nash_2, B, C, p=0.5, N=20)
    grad_nash_3 = calc_gradient(Ck_nash_3_expand, nash_3_Xnk, B, C, nash_3_Xk, Nash_flag=True, beta=beta_3)

    beta_4 = compute_grad_p(nash_4_Xk, Ck_nash_4, B, C, p=2, N=20)
    grad_nash_4 = calc_gradient(Ck_nash_4_expand, nash_4_Xnk, B, C, nash_4_Xk, Nash_flag=True, beta=beta_4)

    beta_5 = compute_grad_p(nash_5_Xk, Ck_nash_5, B, C, p=0.8, N=20)
    grad_nash_5 = calc_gradient(Ck_nash_5_expand, nash_5_Xnk, B, C, nash_5_Xk, Nash_flag=True, beta=beta_5)

    beta_6 = compute_grad_p(nash_6_Xk, Ck_nash_6, B, C, p=0.7, N=20)
    grad_nash_6 = calc_gradient(Ck_nash_6_expand, nash_6_Xnk, B, C, nash_6_Xk, Nash_flag=True, beta=beta_6)

    # Update action of players
    global_Xnk = global_Xnk - lr[t] * grad_global
    nash_Xnk = nash_Xnk - lr[t] * grad_nash
    nash_1_Xnk = nash_1_Xnk - lr[t] * grad_nash_1
    nash_2_Xnk = nash_2_Xnk - lr[t] * grad_nash_2
    nash_3_Xnk = nash_3_Xnk - lr[t] * grad_nash_3
    nash_4_Xnk = nash_4_Xnk - lr[t] * grad_nash_4
    nash_5_Xnk = nash_5_Xnk - lr[t] * grad_nash_5
    nash_6_Xnk = nash_6_Xnk - lr[t] * grad_nash_6
    # Project gradient to constraint
    global_Xnk = project_onto_simplex(global_Xnk)
    nash_Xnk = project_onto_simplex(nash_Xnk)
    nash_1_Xnk = project_onto_simplex(nash_1_Xnk)
    nash_2_Xnk = project_onto_simplex(nash_2_Xnk)
    nash_3_Xnk = project_onto_simplex(nash_3_Xnk)
    nash_4_Xnk = project_onto_simplex(nash_4_Xnk)
    nash_5_Xnk = project_onto_simplex(nash_5_Xnk)
    nash_6_Xnk = project_onto_simplex(nash_6_Xnk)



# Plot results
t = np.arange(T)
plt.figure(1)
plt.plot(t, np.sum(global_record, axis=1), '--k', label="Global")
plt.plot(t, np.sum(nash_record, axis=1), '--r', label="Nash")
plt.plot(t, np.sum(nash_1_record, axis=1), label="P=0.2")
plt.plot(t, np.sum(nash_2_record, axis=1), label="P=0.3")
plt.plot(t, np.sum(nash_3_record, axis=1), label="P=0.5")
plt.plot(t, np.sum(nash_4_record, axis=1), label="P=2")
plt.plot(t, np.sum(nash_5_record, axis=1), label="P=0.8")
plt.plot(t, np.sum(nash_6_record, axis=1), label="P=0.7")
plt.xlabel("# Iterations"), plt.ylabel("# Score"), plt.legend()
plt.ylim(int(np.sum(global_record, axis=1)[T - 1] - 600), int(np.sum(nash_record, axis=1)[T - 1] + 600))
plt.show()


print(f"Global_std = {np.std(np.sum(global_record, axis=1))}")
print(f"Nash_std = {np.std(np.sum(nash_record, axis=1))}")
print(f"P=0.2 std = {np.std(np.sum(nash_1_record, axis=1))}")
print(f"P=0.3, std = {np.std(np.sum(nash_2_record, axis=1))}")
print(f"P=0.5, std={np.std(np.sum(nash_3_record, axis=1))}")
print(f"P=2, std= {np.std(np.sum(nash_4_record, axis=1))}")
print(f"P=0.8, std= {np.std(np.sum(nash_5_record, axis=1))}")
print(f"P=0.7, std= {np.std(np.sum(nash_6_record, axis=1))}")
print("Finsh !!!")

