import numpy as np
import matplotlib.pyplot as plt
import time

def max_to_one(grid):
    max_indices = np.argmax(grid, axis=1)
    modified_grid = np.zeros_like(grid)
    
    for row_idx, max_idx in enumerate(max_indices):
        modified_grid[row_idx, max_idx] = 1
        
    return modified_grid

def normalize_array(arr):
    """
    Normalize a NumPy array by dividing each element by the sum of all elements.

    :param arr: The input NumPy array to be normalized.
    :return: A normalized NumPy array.
    """
    # Find the minimum and maximum values in the array
    arr_min = np.min(arr)
    arr_max = np.max(arr)
    
    # Normalize the array to between 0 and 1
    normalized_arr = (arr - arr_min) / (arr_max - arr_min)
    
    return normalized_arr

def get_sampled_value(value_set):
    num_action = len(value_set)
    strategy = np.ones(num_action) / num_action
    a = 0
    cum_prob = 0
    r = np.random.rand()

    while True:
        if a > num_action - 2:
            break
        cum_prob += strategy[a]
        if r < cum_prob:
            break
        a += 1

    value = value_set[a]
    return value

def generate_datarate_matrix(M, S):
    datarate = np.zeros((M, S))

    for i in range(M):
        for k in range(S):
            # datarate[i, k] = get_sampled_value(np.arange(12, 101, 12))
            datarate[i, k] = np.random.random()/M

    return datarate

def getGlobalUtility(agent_idx, action_vector, datarate_matrix):
    # Implement the getGlobalUtility function based on your use case
    action_idx = action_vector[agent_idx]
    num_agent = datarate_matrix.shape[0]
    num_action = datarate_matrix.shape[1]

    local_utility_matrix = np.zeros((num_agent, num_action))
    action_vector_new = np.zeros((num_action, num_agent), dtype=int)

    for k in range(num_action):
        action_vector_new[k] = action_vector
    action_vector_new[:, agent_idx] = np.arange(num_action)

    count_k = np.zeros((num_action, num_action))

    for k_row in range(num_action):
        for k_column in range(num_action):
            count_k[k_row, k_column] = np.sum(action_vector_new[k_row, :] == (k_column))

    for i in range(num_agent):
        if i == agent_idx:
            local_utility_matrix[i, :] = datarate_matrix[i, :] / (count_k[action_idx, :] + 1)
            local_utility_matrix[i, action_idx] = datarate_matrix[i, action_idx] / count_k[action_idx, action_idx]
        else:
            local_utility_matrix[i, :] = datarate_matrix[i, action_vector[i]] / count_k[:, action_vector[i]]

    global_utility_vector = np.sum(local_utility_matrix, axis=0)
    return global_utility_vector

def socialOptimum(num_players, num_moves, iterations, datarate):
    M = num_players
    S = num_moves
    # Load datarate100x15_v2.mat file and assign it to datarate
    # datarate = ...

    N = iterations
    u = np.zeros((S, M))
    a = np.zeros((N, M), dtype=int)
    x = np.zeros((S, M))
    avgP = np.zeros((N, M))
    realP = np.zeros((N, M))
    count = np.zeros((N, S))
    avgcount = np.zeros((N, S))
    no = np.zeros((S, M))
    unRegret = np.zeros((S, M))
    sumPayoff = np.zeros(N)
    xisquare = np.zeros((N, M))
    globalU = 0
    omega = 1

    delta = 0.0001
    gamma = 0.24

    CLOCK = []
    ALL_MOVES = []
    WEIGHTS = []
    REGRETS = []
    ACTIONS = []

    x[:, :] = 1 / S


    for t in range(1,N):
        if t % 1000 == 0:
            print(t)

        for i in range(M):
            temp = np.random.rand()
            j = 0
            k = x[j, i]
            while temp > k and j < S - 1:
                j += 1
                k += x[j, i]
            a[t, i] = j
        ACTIONS.append(a[t])
        no[a[t, :].astype(int), np.arange(M)] += 1

        for j in range(S):
            count[t, j] = np.sum(a[t, :] == j)
            if t == 1:
                avgcount[t, j] = count[t, j]
            else:
                avgcount[t, j] = (avgcount[t - 1, j] * (t - 1) + count[t, j]) / t

        for i in range(M):
            # if a[t, i] != 0:
            temp = getGlobalUtility(i, a[t, :], datarate)
            u[:, i] = temp
            realP[t, i] = u[int(a[t, i]), i]
            if t == 1:
                avgP[t, i] = realP[t, i]
            else:
                avgP[t, i] = (avgP[t - 1, i] * (t - 1) + realP[t, i]) / t
            xisquare[t, i] = realP[t, i] ** 2

        if np.max(u) > globalU:
            globalU = np.max(u)
        omega = u[int(a[t, 0]), 0] / globalU
        # print(omega)
        WEIGHTS.append(omega)

        for i in range(M):
            unRegret[:, i] = np.maximum(unRegret[:, i] + u[:, i] - u[a[t, i], i], 0)
        # print(np.sum(unRegret))
        REGRETS.append(unRegret)

        for i in range(M):
            if realP[t, i] >= globalU:
                x[a[t, i], i] = 1
            elif np.sum(unRegret[:, i]) == 0:
                x[:, i] = 1 / M

            else:
                x[:, i] = (1 - delta / t ** gamma) * unRegret[:, i] / np.sum(unRegret[:, i]) + (delta / t ** gamma) / M
        # fairness[t, 0] = (np.sum(realP[t, :])) ** 2 / np.sum(xisquare[t, :]) / M
        sumPayoff[t] = np.sum(realP[t])
        # col_sums = np.sum(x, axis=0)
        # zero_sum_columns = np.where(col_sums == 0)[0]
        
        # if zero_sum_columns.size > 0:
        #     print("zero sum")
        CLOCK.append(time.time())
        ALL_MOVES.append(max_to_one(x.T))
    # print(ALL_MOVES[0])
    # Figure 1
    # plt.figure(1)
    # plt.plot(range(1, N), REGRETS, linewidth=3)
    # # plt.plot(range(1, N), normalize_array(REGRETS), linewidth=3)
    # plt.ylabel('Average payoff')
    # plt.xlabel('Iteration')
    # plt.show()

    # # # Figure 2
    # plt.figure(2)
    # plt.plot(range(1, N), realP[1:], linewidth=3)
    # plt.ylabel('Real payoff')
    # plt.xlabel('Iteration')
    # plt.show()

    # # # Figure 3
    # plt.figure(3)
    # plt.plot(range(1, N), count[1:], linewidth=3)
    # plt.ylabel('Num. players per resource')
    # plt.xlabel('Number of iterations')
    # plt.grid()
    # plt.show()

    # # # Figure 4
    # plt.figure(4)
    # plt.plot(range(1, N), avgcount[1:], linewidth=3)
    # plt.ylabel('Average number of users connecting to each networks')
    # plt.xlabel('Iteration')
    # plt.show()

    # # # Figure 5
    # plt.figure(5)
    # plt.plot(range(1, N), sumPayoff[1:] / 100, '-x', label='sumPayoff')
    # plt.ylabel('Total payoffs of all users')
    # plt.xlabel('Iteration')
    # plt.legend()
    # plt.show()
    return np.array(ALL_MOVES), np.array(REGRETS[1:]), WEIGHTS, CLOCK, sumPayoff[1:]

def main():
    # number of players
    M = 3
    # number of actions
    S = 3
    datarate = generate_datarate_matrix(M, S)
    socialOptimum(M, S, 10, datarate)



if __name__ == '__main__':
    main()