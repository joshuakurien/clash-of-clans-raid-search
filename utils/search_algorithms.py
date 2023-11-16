"""
This is an implementation of metaheuristic optimization algorithms.
"""

import random
import numpy as np
from tqdm import tqdm
from typing import Tuple, List, Callable, Optional
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod


##############################################################################################################
############ Local Search (Hill Climbing) Algorithm ##########################################################
##############################################################################################################
def local_search(
    cost_function: Callable,
    max_itr: int,
    convergence_threshold: float,
    x_initial: Optional[np.array] = None,
    x_range: Optional[List[List[float]]] = None,
    hide_progress_bar: Optional[bool] = False,
) -> Tuple[np.array, float, List[np.array], List[float]]:
    # Set the x_initial
    if x_initial is None:
        x_initial = [
            random.uniform(x_range[i][0], x_range[i][1]) for i in range(len(x_range))
        ]

    x_current = x_initial
    cost_current = cost_function(x_current)

    x_history = [x_current]
    cost_history = [cost_current]

    # Create a tqdm progress bar
    if not hide_progress_bar:
        progress_bar = tqdm(total=max_itr, desc="Iterations")

    convergence = False
    itr = 0
    while not convergence:
        # Generate neighboring solutions
        x_neighbor = [random.gauss(x, 0.1) for x in x_current]
        x_neighbor = bound_solution_in_x_range(x=x_neighbor, x_range=x_range)
        cost_neighbor = cost_function(x_neighbor)

        # Accept the neighbor if it has lower cost
        if cost_neighbor < cost_current:
            x_current = x_neighbor
            cost_current = cost_neighbor
            if (cost_current < convergence_threshold) or (itr >= max_itr):
                convergence = True

        x_history.append(x_current)
        cost_history.append(cost_current)

        # Update the tqdm progress bar
        if not hide_progress_bar:
            progress_bar.update(1)  # Increment the progress bar by 1 unit
        itr += 1

    # progress_bar.close()

    # Get the best solution
    best_cost_index = np.argmin(cost_history)
    best_x = x_history[best_cost_index]
    best_cost = cost_history[best_cost_index]

    return best_x, best_cost, x_history, cost_history

def variable_neighborhood_search(cost_function: Callable, max_itr_ils: int, max_itr_ls: int, convergence_threshold: float, d: int,
                           x_range: Optional[List[List[float]]] = None) -> Tuple[np.array, float, List[np.array], List[float]]:
    global_best_cost_index = -1
    global_best_x = -1
    global_best_cost = float('inf')

    # Split each dimension into K intervals
    K = 2
    # Note there will be k**d number of neighborhoods based on the K split we perform here
    # Create ranges of x intervals based on the K split
    x_intervals = []
    for dim_range in x_range:
        lower_bound, upper_bound = dim_range
        interval_size = (upper_bound - lower_bound) / K
        intervals = np.arange(lower_bound, upper_bound + interval_size, interval_size)
        x_intervals.append(intervals)

    # Generate neighborhoods with different bounds for searching
    neighborhoods = []
    for indices in np.ndindex(*([K] * d)):
        bounds = []
        for dim, index in enumerate(indices):
            lower_bound, upper_bound = x_intervals[dim][index], x_intervals[dim][index + 1]
            bounds.append([lower_bound, upper_bound])
        neighborhood = bounds
        neighborhoods.append(neighborhood)

    neighborhood_index = 0

    x_history = []
    cost_history = []
    
    while neighborhood_index < len(neighborhoods):
        neighborhood = neighborhoods[neighborhood_index]
        # Find random point in the neighborhood
        x_current = [random.uniform(neighborhood[i][0], neighborhood[i][1]) for i in range(len(neighborhood))]
        
        # Do local search
        print(f"Searching x:{x_current}")
        best_x, best_cost, _, _ = local_search(cost_function=cost_function, max_itr=max_itr_ls, convergence_threshold=convergence_threshold,
                                            x_initial=x_current, x_range=neighborhood)
        x_history.append(best_x)
        cost_history.append(best_cost)

        # If we find a cost better than the global one, we restart our search at n=0
        # Otherwise we check the next neighbor at n+1
        if best_cost < global_best_cost:
            print(f"Cost:{global_best_cost}")
            global_best_cost = best_cost
            global_best_x = best_x
            neighborhood_index = 0
        else:
            neighborhood_index+=1

    # Get the best solution
    global_best_cost_index = np.argmin(cost_history)
    global_best_x = x_history[global_best_cost_index]
    global_best_cost = cost_history[global_best_cost_index]

    return global_best_x, global_best_cost, x_history, cost_history

##############################################################################################################
############ Helper Functions ################################################################################
##############################################################################################################
def bound_solution_in_x_range(x: List[float], x_range: List[List[float]]) -> List[float]:
    for j in range(len(x)):
        if x[j] < x_range[j][0]:
            x[j] = x_range[j][0]
        elif x[j] > x_range[j][1]:
            x[j] = x_range[j][1]
    return x