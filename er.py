import numpy as np
import time
import seaborn as sns
import scipy.special
import math
import os
import functools
import itertools
import matplotlib.pyplot as plt
from collections import defaultdict
import random
from IPython import embed
from tqdm import tqdm
import argparse

import util
import optim
from game import Game

from socialOptimumRegret import generate_datarate_matrix, getGlobalUtility, max_to_one, get_sampled_value

# Global constants
RUN_REGRET_TESTS = False
EPSILON = 1e-15
regrets_to_plot =  defaultdict(lambda: [])
bars = np.linspace(0, 100, 500)

def socialOptimum(num_players, num_moves, iterations, datarate, game):
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

    CLOCK = [time.time()]
    ALL_MOVES = []
    WEIGHTS = []

    x[:, :] = 1 / S

    last_moves = np.zeros((game.num_players, game.num_moves))

    SWAP, payback = update_regrets(np.zeros((game.num_players, game.num_moves)), game, last_moves)
    EQUILIBRIUM_SWAP, _ = update_regrets(np.zeros((game.num_players, game.num_moves)), game, last_moves)

    REGRET = [total_regrets(SWAP, game)]

    total_weight, WEIGHTS = 1, np.array([1.])
    

    for t in tqdm(range(2, N), disable=False):
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
        

        for i in range(M):
            unRegret[:, i] = np.maximum(unRegret[:, i] + u[:, i] - u[a[t, i], i], 0)
        REGRET.append(unRegret.T)
        # print(unRegret.T)
        # print(np.sum(unRegret))

        for i in range(M):
            if realP[t, i] >= globalU:
                x[a[t, i], i] = 1
            elif np.sum(unRegret[:, i]) == 0:
                x[:, i] = 1 / M

            else:
                x[:, i] = (1 - delta / t ** gamma) * unRegret[:, i] / np.sum(unRegret[:, i]) + (delta / t ** gamma) / M
        # fairness[t, 0] = (np.sum(realP[t, :])) ** 2 / np.sum(xisquare[t, :]) / M
        sumPayoff[t] = np.sum(realP[t])

        last_moves = x.T
        ALL_MOVES.append(np.copy(max_to_one(x.T)))
        # arr = []
        # candidates = np.logspace(-1, 2*np.log(t), 10, base=np.e)
        # for w in candidates:

        #     SWAP_tmp, _ = update_regrets(np.copy(SWAP), game, last_moves, weight=w)
        #     arr.append(total_regrets(SWAP_tmp, game) / (total_weight + w))

        # best_w = candidates[np.argmin(arr)]
        # best_w = 1
        # dilution = 1
        # SWAP, _ = update_regrets(SWAP, game, last_moves, weight=best_w, cfrplus=False, dilution=dilution, only_player=None, payback=payback, optimism=True)
        # total_weight = total_weight / dilution + best_w
        # WEIGHTS *= 1./dilution
        # WEIGHTS = np.append(WEIGHTS, best_w)
        # EQUILIBRIUM_SWAP, _ = update_regrets(np.zeros((game.num_players, game.num_moves)), game, last_moves, weight=best_w, dilution=dilution)
        # REGRET.append(total_regrets(EQUILIBRIUM_SWAP, game) / total_weight)
        # CLOCK.append(time.time())
        # if REGRET[-1] < 0.0:
        #     break
    return np.array(ALL_MOVES), np.array(REGRET), WEIGHTS, CLOCK, sumPayoff[1:]
    
def sweep_floor(game, use_pure_strategies, iterations, seed):

    uniform, step = np.ones(iterations-1), np.arange(1, iterations)
    cfr_equilibrium, cfr_regrets, _, cfr_time = external_regret(game, seed=seed, use_pure_strategies=use_pure_strategies, iterations=iterations)
    linear_equilibrium, linear_regrets, _, linear_time = external_regret(game, seed=seed, use_pure_strategies=use_pure_strategies, iterations=iterations, linear=True)

    for floor in reversed([1./EPSILON, 100, 10, 2, 1, 0.5]):
        exact_equilibrium, exact_regrets, exact_weights, exact_time = external_regret(game, seed=seed, use_pure_strategies=use_pure_strategies, iterations=iterations, dynamic=True, use_potential_function=True, floor=True, floor_amount=floor)
        plot_regrets(exact_equilibrium, exact_regrets, game, exact_weights, "Floor is {}% of average".format(1/floor * 100), exact_time)

    plot_regrets(cfr_equilibrium, cfr_regrets, game, step, "Vanilla RM", cfr_time)
    plot_regrets(linear_equilibrium, linear_regrets, game, step, "Linear RM", linear_time)

def find_greedy(equilibrium, game):

    SWAP = np.zeros((game.num_players, game.num_moves))
    total_weight = 1
    regret_over_time, weights = [], np.array([])
    for t, moves in tqdm(enumerate(equilibrium)):

        utility = game.payoff(moves)
        NXT_SWAP, _ = update_regrets(np.zeros((game.num_players, game.num_moves)), game, moves, weight=1)

        if t == 0:
            best_w = 1
        else:
            R, r = [], []
            for player in range(game.num_players):
                for action in range(game.num_moves):
                    R.append(SWAP[player][action])
                    r.append(NXT_SWAP[player][action])

            best_w, best_phi = optim.find_optimal_weight(np.array(R), np.array(r), total_weight)

            best_w = max(best_w, 0.1)

        weights = np.append(weights, best_w)
        total_weight += best_w
        SWAP += best_w * NXT_SWAP
        
        weights *= ((t+1)/total_weight)
        SWAP *= ((t+1)/total_weight)
        total_weight = t+1

        regret_over_time.append(total_regrets(SWAP, game) / total_weight)

    return weights, np.array(regret_over_time)

# Finalize and save the plot with the relevant experimental parameters
def finalize_plot(experiment_name, iterations, zero_sum, use_pure_strategies, game, seed, ax, game_reps):
    minlength = 1e10
    for name in regrets_to_plot:
        for i in range(len(regrets_to_plot[name])):
            minlength = min(minlength, len(regrets_to_plot[name][i]))

    print("minlength is {}".format(minlength))

    ax.plot(1/np.sqrt(np.arange(1, iterations)), label="1/sqrt(t)")
    for name in regrets_to_plot:
        if "Greedy" in name or "Social Optimum" in name:
            linewidth = 3
        else:
            linewidth = 1
        if args.use_time:
            for i in range(len(regrets_to_plot[name])):
                regrets_to_plot[name][i] = regrets_to_plot[name][i][:minlength]
            avg = np.mean(np.array(regrets_to_plot[name]), axis=0)
            avg[avg < EPSILON] = EPSILON
            std = 1.96 * np.std(np.array(regrets_to_plot[name]), axis=0) / np.sqrt(game_reps)

            ax.errorbar(bars[:len(avg)], avg, label=name, linewidth=linewidth, yerr=std)
        else:
            avg = np.mean(np.array(regrets_to_plot[name]), axis=0)
            avg[avg < EPSILON] = EPSILON
            std = 1.96 * np.std(np.array(regrets_to_plot[name]), axis=0) / np.sqrt(game_reps)

            ax.errorbar(np.arange(1, len(avg)+1), avg, label=name, linewidth=linewidth, yerr=std)

    fpath = "figs/{}_{}".format(experiment_name, args.use_time)
    if not os.path.exists(fpath):
        os.mkdir(fpath)

    plt.xscale("log"), plt.yscale("log")

    game_type = "zero" if zero_sum else "general"
    strategy = "pure" if use_pure_strategies else "mixed"
    game_name = "External Regret Minimization for {}-player {}-action {}-sum games".format(game.num_players, game.num_moves, game_type)

    plt.title(game_name)
    if args.use_time:
        plt.xlabel("Seconds (logscale)")
    else:
        plt.xlabel("Iterations (logscale)")
    plt.ylabel("Regret (logscale)")
    plt.legend(loc=3, prop={'size': 8})
    fname = "{}/{}P-{}A-{}-{}-{}I-{}S-EXTERNAL".format(fpath, game.num_players, game.num_moves, game_type, strategy, iterations, seed)
    plt.savefig(fname + ".pdf", dpi=1000)

    with open(fname + ".json", "w+") as f:
        f.write(str(regrets_to_plot))

    plt.show()

def finalize_plot_utility(experiment_name, iterations, zero_sum, use_pure_strategies, game, seed, ax, game_reps):
    minlength = 1e10
    for name in regrets_to_plot:
        for i in range(len(regrets_to_plot[name])):
            minlength = min(minlength, len(regrets_to_plot[name][i]))

    print("minlength is {}".format(minlength))

    # ax.plot(normalize_array(1/np.sqrt(np.arange(1, iterations))), label="1/sqrt(t)")
    for name in regrets_to_plot:
        if "Social Optimum" in name:
            linewidth = 3
        elif "Greedy" in name:
            linewidth = 1
        else:
            linewidth = 1
        if args.use_time:
            for i in range(len(regrets_to_plot[name])):
                regrets_to_plot[name][i] = regrets_to_plot[name][i][:minlength]
            avg = np.mean(np.array(regrets_to_plot[name]), axis=0)
            avg[avg < EPSILON] = EPSILON
            std = 1.96 * np.std(np.array(regrets_to_plot[name]), axis=0) / np.sqrt(game_reps)

            ax.errorbar(bars[:len(avg)], avg, label=name, linewidth=linewidth, yerr=std)
        else:
            avg = np.mean(np.array(regrets_to_plot[name]), axis=0)
            avg[avg < EPSILON] = EPSILON
            std = 1.96 * np.std(np.array(regrets_to_plot[name]), axis=0) / np.sqrt(game_reps)

            ax.errorbar(np.arange(1, len(avg)+1), avg, label=name, linewidth=linewidth, yerr=std)

    fpath = "figs/{}_{}".format(experiment_name, args.use_time)
    if not os.path.exists(fpath):
        os.mkdir(fpath)

    plt.xscale("log"), plt.yscale("log")

    game_type = "zero" if zero_sum else "general"
    strategy = "pure" if use_pure_strategies else "mixed"
    game_name = "Average Payoff Over Time for {}-player {}-action {}-sum games".format(game.num_players, game.num_moves, game_type)

    plt.title(game_name)
    if args.use_time:
        plt.xlabel("Seconds")
    else:
        plt.xlabel("Iterations")
    plt.ylabel("Payoff")
    plt.legend(loc=3, prop={'size': 8})
    fname = "{}/{}P-{}A-{}-{}-{}I-{}S-EXTERNAL".format(fpath, game.num_players, game.num_moves, game_type, strategy, iterations, seed)
    plt.savefig(fname + ".pdf", dpi=1000)

    with open(fname + ".json", "w+") as f:
        f.write(str(regrets_to_plot))

    plt.show()

# Save the regrets to be plotted
def plot_regrets(equilibrium, regrets, game, weights, name, time):
    time = np.array(time) - min(time)
    if args.use_time:
        marker = 0
        results = []
        for r, t in zip(regrets, time):

            while marker < len(bars) and t > bars[marker]:
                results.append(r)
                marker += 1

            if marker >= len(bars):
                break

        results = np.array(results)
        regrets_to_plot[name].append(np.copy(results))
    else:
        regrets = manual_regret(equilibrium, game, weights)
        # if RUN_REGRET_TESTS:
        #     assert np.max(np.abs(regrets - all_regrets)) < 1e-10, "Potentially a mistake calculating regret (or over/underflow in the testing function)"
        regrets_to_plot[name].append(regrets)

def plot_utility(equilibrium, game, name):
    utilities = manual_utility(equilibrium, game)
    regrets_to_plot[name].append(utilities)


def plot_regrets_socialOptimum(equilibrium, regrets, game, weights, name, time):
        SWAP = np.zeros((game.num_players, game.num_moves))
        regret_over_time = []
        total_weight = 0
        utility_over_time = []
        for t, moves in tqdm(enumerate(equilibrium)):

            utility = game.payoff(moves)
            utility_over_time.append(np.sum(utility))

            total_weight += weights[t]

            for player in range(game.num_players):
                player_regrets = util.regret(game, moves, player)
                SWAP[player, :] += weights[t] * player_regrets
            
            cur_regret = total_regrets(SWAP, game)
            regret_over_time.append(cur_regret / total_weight)
    
        regrets_to_plot[name].append((np.array(regret_over_time)))

def plot_SO_utility(utility_over_time, name):
    regrets_to_plot[name].append((np.array(utility_over_time)))


def benchmark_time(game, iterations, seed):

    uniform, step, square = np.ones(iterations-1), np.arange(1, iterations), np.square(np.arange(1, iterations))
    opt_base_equilibrium, opt_base_regrets, _, plus_time = external_regret(game, seed=seed, use_pure_strategies=False, iterations=iterations, cfrplus=True, optimism=True)
    exact_equilibrium, exact_regrets, exact_weights, exact_time = external_regret(game, seed=seed, use_pure_strategies=True, iterations=iterations, dynamic=True, use_potential_function=True, floor=True, floor_amount=2)
    plot_regrets(opt_base_equilibrium, opt_base_regrets, game, step, "Optimistic RM+ (mixed)", plus_time)
    plot_regrets(exact_equilibrium, exact_regrets, game, exact_weights, "Greedy Weights (pure)", exact_time)

# Sweep through all regret minimization methods except dynamic weights
def full_sweep(game, use_pure_strategies, iterations, seed):

    # Weighting schemes
    uniform, step = np.ones(iterations-1), np.arange(1, iterations)
    plot_cfr, plot_plus, plot_linear, plot_dynamic, plot_socialOptimum = True, True, True, True, True

    if plot_cfr:
        print("RM")
        cfr_equilibrium, cfr_regrets, _, cfr_time = external_regret(game, seed=seed, use_pure_strategies=use_pure_strategies, iterations=iterations)
        #alt_cfr_equilibrium, alt_cfr_regrets, _ = external_regret(game, seed=seed, use_pure_strategies=use_pure_strategies, iterations=iterations, alternating=True)
        #opt_cfr_equilibrium, opt_cfr_regrets, _ = external_regret(game, seed=seed, use_pure_strategies=use_pure_strategies, iterations=iterations, optimism=True)

        plot_regrets(cfr_equilibrium, cfr_regrets, game, uniform, "RM", cfr_time)
        #plot_regrets(alt_cfr_equilibrium, alt_cfr_regrets, game, uniform, "RM w/ Alternating Updates")
        #plot_regrets(opt_cfr_equilibrium, opt_cfr_regrets, game, uniform, "RM w/ Optimism")

    if plot_plus:
        print("RM+")
        #plus_equilibrium, plus_regrets, _ = external_regret(game, seed=seed, use_pure_strategies=use_pure_strategies, iterations=iterations, cfrplus=True)
        #alt_plus_equilibrium, alt_plus_regrets, _ = external_regret(game, seed=seed, use_pure_strategies=use_pure_strategies, iterations=iterations, alternating=True, cfrplus=True)
        opt_plus_equilibrium, opt_plus_regrets, _, opt_plus_time = external_regret(game, seed=seed, use_pure_strategies=use_pure_strategies, iterations=iterations, optimism=True, cfrplus=True)

        #plot_regrets(plus_equilibrium, plus_regrets, game, step, "RM+")
        #plot_regrets(alt_plus_equilibrium, alt_plus_regrets, game, step, "RM+ w/ Alternating Updates")
        plot_regrets(opt_plus_equilibrium, opt_plus_regrets, game, step, "RM+ w/ Optimism", opt_plus_time)

    if plot_linear:
        print("Linear RM")
        linear_equilibrium, linear_regrets, _, linear_time = external_regret(game, seed=seed, use_pure_strategies=use_pure_strategies, iterations=iterations, linear=True)
        print("Linear RM w/ Alternating Updates")
        alt_linear_equilibrium, alt_linear_regrets, _, alt_linear_time = external_regret(game, seed=seed, use_pure_strategies=use_pure_strategies, iterations=iterations, alternating=True, linear=True)
        print("Linear RM w/ Optimism")
        opt_linear_equilibrium, opt_linear_regrets, _, opt_linear_time = external_regret(game, seed=seed, use_pure_strategies=use_pure_strategies, iterations=iterations, optimism=True, linear=True)

        plot_regrets(linear_equilibrium, linear_regrets, game, step, "Linear RM", linear_time)
        plot_regrets(alt_linear_equilibrium, alt_linear_regrets, game, step, "Linear RM w/ Alterating Updates", alt_linear_time)
        plot_regrets(opt_linear_equilibrium, opt_linear_regrets, game, step,
                     "Linear RM w/ Optimism", opt_linear_time)

    if plot_dynamic:
        print("Greedy Weights")
        pot_equilibrium, pot_regrets, pot_weights, pot_time = external_regret(game, seed=seed, use_pure_strategies=use_pure_strategies, iterations=iterations, dynamic=True, use_potential_function=True)

        plot_regrets(pot_equilibrium, pot_regrets, game, pot_weights, "Greedy Weights", pot_time)


    if plot_socialOptimum:
        print("Social Optimum")
        datarate = generate_datarate_matrix(game.num_players, game.num_moves)
        social_equilibrium, social_regrets, social_weights, social_time, social_utility = socialOptimum(game.num_players, game.num_moves, iterations=iterations, datarate=datarate, game=game)
        social_equilibrium, social_regrets, social_weights, social_time = external_regret(game, seed=seed, use_pure_strategies=use_pure_strategies, iterations=iterations, social_optimum=True, so_equilibrium=social_equilibrium, optimism=True, linear=True)
        plot_regrets(social_equilibrium, social_regrets, game, social_weights, "Social Optimum", social_time)

def full_utility(game, use_pure_strategies, iterations, seed):
    # Weighting schemes
    uniform, step = np.ones(iterations-1), np.arange(1, iterations)
    plot_cfr, plot_plus, plot_linear, plot_dynamic, plot_socialOptimum = True, True, True, True, True


    if plot_cfr:
        cfr_equilibrium, cfr_regrets, _, cfr_time = external_regret(game, seed=seed, use_pure_strategies=use_pure_strategies, iterations=iterations)
        plot_utility(cfr_equilibrium, game, "RM")

    if plot_plus:

        #plus_equilibrium, plus_regrets, _ = external_regret(game, seed=seed, use_pure_strategies=use_pure_strategies, iterations=iterations, cfrplus=True)
        #alt_plus_equilibrium, alt_plus_regrets, _ = external_regret(game, seed=seed, use_pure_strategies=use_pure_strategies, iterations=iterations, alternating=True, cfrplus=True)
        opt_plus_equilibrium, opt_plus_regrets, _, opt_plus_time = external_regret(game, seed=seed, use_pure_strategies=use_pure_strategies, iterations=iterations, optimism=True, cfrplus=True)

        #plot_regrets(plus_equilibrium, plus_regrets, game, step, "RM+")
        #plot_regrets(alt_plus_equilibrium, alt_plus_regrets, game, step, "RM+ w/ Alternating Updates")
        plot_utility(opt_plus_equilibrium, game, "RM+ w/ Optimism")

    if plot_linear:

        linear_equilibrium, linear_regrets, _, linear_time = external_regret(game, seed=seed, use_pure_strategies=use_pure_strategies, iterations=iterations, linear=True)
        alt_linear_equilibrium, alt_linear_regrets, _, alt_linear_time = external_regret(game, seed=seed, use_pure_strategies=use_pure_strategies, iterations=iterations, alternating=True, linear=True)
        opt_linear_equilibrium, opt_linear_regrets, _, opt_linear_time = external_regret(game, seed=seed, use_pure_strategies=use_pure_strategies, iterations=iterations, optimism=True, linear=True)

        # plot_regrets(linear_equilibrium, linear_regrets, game, step, "Linear RM", linear_time)
        # plot_regrets(alt_linear_equilibrium, alt_linear_regrets, game, step, "Linear RM w/ Alterating Updates", alt_linear_time)
        plot_utility(linear_equilibrium, game, "Linear RM")
        plot_utility(alt_linear_equilibrium, game, "Linear RM w/ Alterating Updates")
        # plot_regrets(opt_linear_equilibrium, opt_linear_regrets, game, step,
        #              "Linear RM w/ Optimism", opt_linear_time)
        plot_utility(opt_linear_equilibrium, game, "Linear RM w/ Optimism")

    if plot_dynamic:
        pot_equilibrium, pot_regrets, pot_weights, pot_time = external_regret(game, seed=seed, use_pure_strategies=use_pure_strategies, iterations=iterations, dynamic=True, use_potential_function=True)
        # plot_regrets(pot_equilibrium, pot_regrets, game, pot_weights, "Greedy Weights", pot_time)
        plot_utility(pot_equilibrium, game, "Greedy Weights")


    if plot_socialOptimum:

        datarate = generate_datarate_matrix(game.num_players, game.num_moves)
        social_equilibrium, social_regrets, social_weights, social_time, social_utility = socialOptimum(game.num_players, game.num_moves, iterations=iterations, datarate=datarate, game=game)
        # plot_regrets_socialOptimum(social_equilibrium, social_regrets, game, social_weights, "Social Optimum", social_time)
        plot_utility(social_equilibrium, game, "Social Optimum")
        # plot_SO_utility(social_utility, "Social Optimum")


# Compare variations of dynamic weights to each other
def compare_dynamic(game, use_pure_strategies, iterations, seed):

    dyn_equilibrium, dyn_regrets, dyn_weights, dyn_time = external_regret(game, use_pure_strategies=True, iterations=iterations, dynamic=True, use_optimize=False, use_potential_function=False)
    pot_equilibrium, pot_regrets, pot_weights, pot_time = external_regret(game, use_pure_strategies=True, iterations=iterations, dynamic=True, use_potential_function=True)
    sweep_equilibrium, sweep_regrets, sweep_weights, sweep_time = external_regret(game, use_pure_strategies=True, iterations=iterations, dynamic=True, use_optimize=False)
    max_equilibrium, max_regrets, max_weights, max_time = external_regret(game, use_pure_strategies=True, iterations=iterations, dynamic=True, use_max=True, use_optimize=False, use_potential_function=False)
    floor_equilibrium, floor_regrets, floor_weights, floor_time = external_regret(game, use_pure_strategies=True, iterations=iterations, dynamic=True, floor=True)


    plot_regrets(dyn_equilibrium, dyn_regrets, game, dyn_weights, "Dynamic (sum)", dyn_time)
    plot_regrets(pot_equilibrium, pot_regrets, game, pot_weights, "Dynamic (potential)", pot_time)
    plot_regrets(max_equilibrium, max_regrets, game, max_weights, "Dynamic (max)", max_time)
    plot_regrets(floor_equilibrium, floor_regrets, game, floor_weights, "Dynamic (floor)", floor_time)
    plot_regrets(sweep_equilibrium, sweep_regrets, game, sweep_weights, "Dynamic (sweep)", sweep_time)

# Update regrets given regret minimization settings and the last moves played
def update_regrets(SWAP, game, last_moves, weight=1, cfrplus=False, dilution=1, only_player=None, payback={}, optimism=False):

    assert SWAP.shape == (game.num_players, game.num_moves), "Wrong SWAP shape"

    if bool(payback):
        assert optimism, "Payback should only be nonempty if optimism is True"

    SWAP *= 1./dilution
    new_payback = {}

    for player in range(game.num_players):

        if only_player is not None and player != only_player:
            continue

        player_regret = weight * util.regret(game, last_moves, player)

        if player in payback:
            SWAP[player, :] -= payback[player]

        SWAP[player, :] += player_regret

        # Optimism adds it twice and then subtracts it out later
        if optimism:
            SWAP[player, :] += player_regret
            new_payback[player] = player_regret

    if cfrplus:
        SWAP[SWAP<0] = 0

    return SWAP, new_payback

# Computes the SWAP regret manually to double check for bugs
def manual_regret(equilibrium, game, weights):
    # print(weights)
    SWAP = np.zeros((game.num_players, game.num_moves))
    regret_over_time = []
    total_weight = 0

    for t, moves in tqdm(enumerate(equilibrium)):

        utility = game.payoff(moves)
        total_weight += weights[t]

        for player in range(game.num_players):

            player_regrets = util.regret(game, moves, player)
            SWAP[player, :] += weights[t] * player_regrets
   
        cur_regret = total_regrets(SWAP, game)
        regret_over_time.append(cur_regret / total_weight)
    return np.array(regret_over_time)

def manual_utility(eq, game):
    utility_over_time = []
    total_utility = 0
    for t, moves in tqdm(enumerate(eq)):
        utility = game.payoff(moves)
        total_utility += np.sum(utility)
        # utility_over_time.append(np.sum(utility)/(game.num_players))
        utility_over_time.append(total_utility/(t+1))
    return np.array(utility_over_time)


def blackwells(SWAP, player, game, restricted_moves):
    regrets = np.array([SWAP[(player, A)] for A in range(game.num_moves)])
    if restricted_moves:
        for move in restricted_moves[player]:
            regrets[move] = 0.0

    regrets[regrets <= 0] = EPSILON
    regrets /= np.sum(regrets)
    return regrets

 # Each player chooses their moves based on past regrets
def compute_next_moves(SWAP, use_pure_strategies, last_moves, game, restricted_moves):

    for player in range(game.num_players):

        probs = blackwells(SWAP, player, game, restricted_moves)
        
        if use_pure_strategies:

            last_moves[player, :] = np.zeros(game.num_moves)
            try:
                last_moves[player][np.random.choice(game.num_moves, p=probs)] = 1
            except:
                embed()

        else:
            last_moves[player, :] = probs

    return last_moves

# Computes external regret
def external_regret(game,
                    iterations,
                    use_pure_strategies,
                    seed=None,
                    so_equilibrium=None,
                    social_optimum=False,
                    # Turn on dynamic weights
                    dynamic=False,
                    # Optimistic updates from Syrgkanis+ 2015
                    optimism=False,
                    # CFR+ from Tammelin 2014
                    cfrplus=False,
                    # Linear CFR from Brown & Sandholm 2019
                    linear=False,
                    # Alternating updates from Tammelin 2014
                    alternating=False,
                    # use the potential function instead of minimizing the regrets
                    use_potential_function=True,
                    # use the max regret as the function to minimize
                    use_max=False,
                    # use optim.py to dtermine the optimal weight
                    use_optimize=True,
                    # don't print the progress bar
                    silent=False,
                    # floor the weights at 2/t * total_weight
                    floor=True,
                    # floor amount,
                    floor_amount=2,
                    # For double oracle, do not allow any moves in the set described
                    restricted_moves=None,
                    # when to quit RM
                    target_epsilon=0.0,
                    # initial move to seed the algorithm,
                    initial_move=None):

    np.random.seed(seed)
    CLOCK = [time.time()]

    if initial_move is None:
        last_moves = np.zeros((game.num_players, game.num_moves))
        for player in range(game.num_players):
            probs = np.ones(game.num_moves) / game.num_moves
            if restricted_moves:
                for move in restricted_moves[player]:
                    probs[move] = 0.0
                    probs /= np.sum(probs)

            if use_pure_strategies:
                last_moves[player, :] = np.zeros(game.num_moves)
                last_moves[player][np.random.choice(game.num_moves, p=probs)] = 1
            else:
                last_moves[player, :] = probs
    else:
        last_moves = np.copy(initial_move)

    # SWAP keeps track of all the regrets. Payback is responsible for "paying back" the optimistic regrets (only if using optimism).
    SWAP, payback = update_regrets(np.zeros((game.num_players, game.num_moves)), game, last_moves)
    EQUILIBRIUM_SWAP, _ = update_regrets(np.zeros((game.num_players, game.num_moves)), game, last_moves)
    # Keeps track of the regret over time.
    REGRET = [total_regrets(SWAP, game)]

    # ALL_MOVES stores the history of plays, which you need to actually compute the correlated equilibrium
    ALL_MOVES = [np.copy(last_moves)]
    total_weight, WEIGHTS = 1, np.array([1.])

    # Used only for the reward dynamic weighting
    past_utility = np.sum(game.payoff(last_moves))

    if not silent:
        print("Starting regret minimization")

    for t in tqdm(range(2, iterations), disable=silent):

        last_moves = np.zeros((game.num_players, game.num_moves))

        last_moves = compute_next_moves(SWAP, use_pure_strategies, last_moves, game, restricted_moves)
        if social_optimum:
            last_moves = compute_next_moves(-so_equilibrium[t-2], use_pure_strategies, last_moves, game, restricted_moves)
 
        if dynamic:
            assert not (alternating and optimism), "Incompatible"

            if alternating:

                assert game.num_players == 2, "Not implemented in general"
                assert not use_pure_strategies, "Useless to sample"
                # DO THIS HERE
                SWAP_tmp, _ = update_regrets(np.copy(SWAP), game, last_moves, weight=total_weight/t, only_player = t % 2)
                if social_optimum:
                    SWAP_tmp, _ = update_regrets(np.copy(-so_equilibrium[t-2]), game, last_moves, weight=total_weight/t, only_player = t % 2)

                nxt_moves = compute_next_moves(SWAP_tmp, use_pure_strategies, np.zeros((game.num_players, game.num_moves)), game, restricted_moves)

                last_moves = (last_moves + nxt_moves) / 2.0

            elif optimism:
                SWAP_tmp, _ = update_regrets(np.copy(SWAP), game, ALL_MOVES[-1], weight=WEIGHTS[-1])
                if social_optimum:
                    SWAP_tmp, _ = update_regrets(np.copy(-so_equilibrium[t-2]), game, ALL_MOVES[-1], weight=WEIGHTS[-1])
                last_moves = compute_next_moves(SWAP_tmp, use_pure_strategies, np.zeros((game.num_players, game.num_moves)), game, restricted_moves)

            # if social_optimum:
            #     last_moves = so_equilibrium[t-2]

            if not use_optimize:
                arr = []
                candidates = np.logspace(-1, 2*np.log(t), 10, base=np.e)
                for w in candidates:

                    SWAP_tmp, _ = update_regrets(np.copy(SWAP), game, last_moves, weight=w)
                    if social_optimum:
                        SWAP_tmp, _ = update_regrets(np.copy(-so_equilibrium[t-2]), game, last_moves, weight=w)
                    arr.append(total_regrets(SWAP_tmp, game) / (total_weight + w))

                best_w = candidates[np.argmin(arr)]

            else:
                best_w = optimize(SWAP, game, last_moves, total_weight, t, restricted_moves)
                if social_optimum:
                    best_w = optimize(-so_equilibrium[t-2], game, last_moves, total_weight, t, restricted_moves)

            if floor:
                best_w = max(best_w, total_weight/(floor_amount*t))

        else:
            best_w = 1

        if linear:
            dilution = t / (t-1)
        else:
            dilution = 1

        if alternating and not dynamic:
            only_player = t % game.num_players
        else:
            only_player = None

        if best_w == np.inf:
            best_w = 1
            dilution = 1000000
        # if social_optimum:
        #     last_moves = so_equilibrium[t-2]
        SWAP, payback = update_regrets(SWAP, game, last_moves, weight=best_w, cfrplus=cfrplus, dilution=dilution, only_player=only_player, payback=payback, optimism=(optimism and not dynamic))
        total_weight = total_weight / dilution + best_w
        WEIGHTS *= 1./dilution
        WEIGHTS = np.append(WEIGHTS, best_w)

        past_utility += best_w * np.sum(game.payoff(last_moves))

        if dynamic or not cfrplus:
            EQUILIBRIUM_SWAP, _ = update_regrets(EQUILIBRIUM_SWAP, game, last_moves, weight=best_w, dilution=dilution)
            if social_optimum:
                EQUILIBRIUM_SWAP, _ = update_regrets(-so_equilibrium[t-2], game, last_moves, weight=best_w, dilution=dilution)

            REGRET.append(total_regrets(EQUILIBRIUM_SWAP, game) / total_weight)
        else:
            EQUILIBRIUM_SWAP, _ = update_regrets(EQUILIBRIUM_SWAP, game, last_moves, weight=t, dilution=dilution)
            if social_optimum:
                EQUILIBRIUM_SWAP, _ = update_regrets(-so_equilibrium[t-2], game, last_moves, weight=t, dilution=dilution)
            REGRET.append(total_regrets(EQUILIBRIUM_SWAP, game) / (t*(t+1)/2))

        ALL_MOVES.append(np.copy(last_moves))
        if restricted_moves:
            restricted_regret = 0.0

            for player in range(game.num_players):
                player_regret = 0.0
                for move in range(game.num_moves):
                    if move in restricted_moves[player]:
                        continue

                    player_regret = max(player_regret, SWAP[player, move])

                restricted_regret += player_regret

            restricted_regret /= total_weight
            #print(restricted_regret)
            if restricted_regret < target_epsilon:
                break
        else:
            if REGRET[-1] < target_epsilon:
                break

        CLOCK.append(time.time())
    return np.array(ALL_MOVES), np.array(REGRET), WEIGHTS, CLOCK

# Dynamically choose the weights to minimize something
def optimize(        SWAP,
                     game,
                     last_moves, 
                     total_weight,
                     timestep,
                     restricted_moves):

    NEW_SWAP, _  = update_regrets(np.zeros(SWAP.shape), game, last_moves, weight=1)

    if restricted_moves:
        R, r = [], []
        for player in range(game.num_players):
            for a in range(game.num_moves):
                if a in restricted_moves[player]:
                    continue

                R.append(SWAP[player][a])
                r.append(NEW_SWAP[player][a])

        R = np.array(R)
        r = np.array(r)
    else:
        R = np.array([SWAP[player][a] for player in range(game.num_players) for a in range(game.num_moves)])
        r = np.array([NEW_SWAP[player][a] for player in range(game.num_players) for a in range(game.num_moves)])

    best_w, best_phi = optim.find_optimal_weight(R, r, total_weight)
    if RUN_REGRET_TESTS and best_w != np.inf:
        SWAP_tmp, _ = update_regrets(np.copy(SWAP), game, last_moves, weight=best_w)
        average_pot = potential_func(SWAP_tmp, game) / ((total_weight + best_w)**2)
        assert abs(best_phi - average_pot) < 1e-8, "With weight {}, {} is not equal to {}".format(best_w, average_pot, best_phi)

    return best_w
    
# Dynamically choose the weights to minimize something
def find_best_weight(dynamic_range,
                     SWAP,
                     game,
                     last_moves, 
                     total_weight,
                     timestep,
                     use_potential_function=False,
                     use_max=False,
                     exploit=False,
                     MIXED_STRAT=None):
    # Sweep from weights of 1 to t^2. Using logspace is CRUCIAL for performance
    candidates = np.logspace(0, np.log(timestep**3), dynamic_range, base=np.e)
    arr = []

    for w in candidates:

        if exploit:
            # Dynamically choose the weight to minimize exploitability

            NEW_MIXED = (np.copy(MIXED_STRAT) + w * last_moves) / (total_weight + w)
            arr.append(util.compute_exploitability(game, NEW_MIXED))
        else:
            SWAP_tmp, _ = update_regrets(np.copy(SWAP), game, last_moves, weight=w)
            if use_potential_function:
                # Minimize the potential function
                arr.append(potential_func(SWAP_tmp, game) / ((total_weight + w)**2))

            elif use_max:
                # Minimize max regret
                arr.append(max_regrets(SWAP_tmp, game) / (total_weight + w))

            else:
                # Minimize the regrets
                arr.append(total_regrets(SWAP_tmp, game) / (total_weight + w))

    return candidates[np.argmin(arr)]
      
# Compute max internal regret for any player
def max_regrets(SWAP, game):
    return max(sum(max(0, np.max(SWAP[player, :])) for player in range(game.num_players)))

# Compute total internal regret for all players
def total_regrets(SWAP, game):
    return sum(max(0, np.max(SWAP[player, :])) for player in range(game.num_players))

# Potential func, which is the squared positive regrets. Should monotonically go decrease.
def potential_func(SWAP, game):
    return np.sum(np.square(SWAP.clip(0)))
              
def main(num_players, num_moves, zero_sum, seed, iterations, use_pure_strategies, experiment_name, game_reps):

    #assert args.full_sweep + args.solo_compare + args.compare_dynamic + args.compare_mixed == 1, "Exactly one option must be specified"

    np.random.seed(seed)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_prop_cycle('color', [plt.get_cmap('tab20')(1. * i / 13) for i in range(13)])

    for i in range(game_reps):
        print("Iteration {} of {}".format(i, game_reps))
        game = Game(num_players=num_players, num_moves=num_moves, zero_sum=zero_sum)

        if args.solo_compare:
            solo_compare(game, use_pure_strategies, iterations, seed)
        elif args.sweep_floor:
            sweep_floor(game, True, iterations, seed)
        elif args.full_sweep:
            full_sweep(game, use_pure_strategies, iterations, seed)
        elif args.compare_dynamic:
            compare_dynamic(game, use_pure_strategies, iterations, seed)
        elif args.compare_mixed:
            compare_mixed(game)
        elif args.use_time:
            benchmark_time(game, iterations, seed)
        elif args.full_utility:
            full_utility(game, use_pure_strategies, iterations, seed)
        else:
            raise Exception("This option does not exist")
    if args.full_utility:
        finalize_plot_utility(experiment_name, iterations, zero_sum, use_pure_strategies, game, seed, ax, game_reps)
    else:
        finalize_plot(experiment_name, iterations, zero_sum, use_pure_strategies, game, seed, ax, game_reps)

def compare_mixed(game):
    cfr_equilibrium, cfr_regrets, _, _ = external_regret(game, seed=None, use_pure_strategies=False, iterations=10000, optimism=False, cfrplus=True)
    weights, regrets = find_greedy(cfr_equilibrium, game)
    plt.plot(cfr_regrets, label="cfr")
    plt.plot(regrets, label="greedy")
    plt.yscale('log')
    plt.legend()
    plt.show()

    quit()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Specify internal regret settings')
    parser.add_argument('--num_moves', type=int, default=4, help="Number of moves in the game, default 3")
    parser.add_argument('--num_players', type=int, default=4, help="Number of players in the game, default 2")
    parser.add_argument('--iterations', type=int, default=10000, help="Number of iterations to run regret minimization, default 1K")
    parser.add_argument('--seed', type=int, default=42, help="Random seed, default 42")
    parser.add_argument('--zero_sum', action="store_true", default=False, help="Make the game zero sum, default false")
    parser.add_argument('--solo_compare', action="store_true", default=False, help="Compare only RM and dynamic weights")
    parser.add_argument('--sweep_floor', action="store_true", default=False, help="Sweep floors")
    parser.add_argument('--compare_dynamic', action="store_true", default=False, help="Compare the variants of dynamic weights")
    parser.add_argument('--full_sweep', action="store_true", default=False, help="Sweep through all non dynamic methods")
    parser.add_argument('--full_utility', action="store_true", default=True, help="Sweep through all non dynamic methods")
    parser.add_argument('--compare_mixed', action="store_true", default=False, help="Compare mixed dynamic")
    parser.add_argument('--use_time', action="store_true", default=False, help="Compare time")
    parser.add_argument('--pure', action="store_true", default=True, help="Use pure strategies for regret minimization, default false")
    parser.add_argument("--experiment_name", type=str, help="What is this experiment called?", default="CYBERTRASH")
    parser.add_argument("--game_reps", type=int, help="How many games to repeat for", default=1)
    args = parser.parse_args()
    print(args)
    main(args.num_players, args.num_moves, args.zero_sum, args.seed, args.iterations, args.pure, args.experiment_name, args.game_reps)
