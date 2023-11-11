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


class BaseParticle:
    def __init__(self):
        self.position = None
        self.velocity = None
        self.best_position = None
        self.best_cost = None
        self.position_history = None
        self.local_best_position = None

        # values that need to be set by the algorithm container
        self.x_range = None
        self.cost_function = None
        self.local_best_option = None

        # particle needs values initialized by container
        self.particle_initialized = False

    def initialize_particle_internals(self):
        assert self.x_range is not None
        assert self.cost_function is not None

        self.position = np.array([random.uniform(x[0], x[1]) for x in self.x_range])
        self.velocity = np.array(
            [random.uniform(-1, 1) for _ in range(len(self.x_range))]
        )
        self.best_position = self.position
        self.best_cost = float("inf")
        self.position_history = []
        self.local_best_position = None

        # set particle intialization flag to True
        self.particle_initialized = True

    def set_x_range(self, x_range: Optional[List[List[float]]]):
        self.x_range = x_range

    def set_cost_function(self, cost_function: Callable):
        self.cost_function = cost_function

    def set_best_position(self, best_position):
        self.best_position = best_position

    def set_local_best_option(self, local_best_option: str):
        self.local_best_option = local_best_option

    def perform_local_search(self, max_itr: int, convergence_threshold: float):
        assert self.particle_initialized
        search_position, search_cost, _, _ = local_search(
            cost_function=self.cost_function,
            max_itr=max_itr,
            convergence_threshold=convergence_threshold,
            x_initial=self.position,
            x_range=self.x_range,
            hide_progress_bar=True,
        )

        # Find the local best particle (for use in the velocity vector):
        if self.local_best_option == "this_iteration":
            self.local_best_position = search_position
        elif self.local_best_option == "so_far":
            if search_cost < self.best_cost:
                self.best_cost = search_cost
                self.best_position = search_position
                self.local_best_position = self.best_position
        else:
            raise ValueError("No local best option selected")

        return search_position, search_cost

    def bound_solution_in_x_range(self, x: List[float]) -> List[float]:
        for j in range(len(x)):
            if x[j] < self.x_range[j][0]:
                x[j] = self.x_range[j][0]
            elif x[j] > self.x_range[j][1]:
                x[j] = self.x_range[j][1]
        return x

    @abstractmethod
    def update_position(self, global_best_position: List[float]):
        pass

    @abstractmethod
    def update_velocity(self, global_best_position: List[float]):
        pass

    def update(self, global_best_position: List[float]):
        # assert self.particle_initalized == True
        self.update_position(global_best_position)
        self.update_velocity(global_best_position)


class DefaultParticle(BaseParticle):
    def __init__(
        self,
        alpha_1: float,
        alpha_2: float,
        alpha_3: float,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.alpha_1 = alpha_1
        self.alpha_2 = alpha_2
        self.alpha_3 = alpha_3

    def update_velocity(self, global_best_position: List[float]):
        self.velocity = (
            self.alpha_1 * self.velocity
            + self.alpha_2 * (self.local_best_position - self.position)
            + self.alpha_3 * (global_best_position - self.position)
        )

    def update_position(self, global_best_position: List[float]):
        self.position = self.position + self.velocity
        self.position = self.bound_solution_in_x_range(x=self.position)
        self.position_history.append(self.position)


class PSO:
    def __init__(
        self,
        cost_function: Callable,
        max_itr: int,
        x_range: Optional[List[List[float]]],
        local_best_option: Optional[str] = "this_iteration",
        global_best_option: Optional[str] = "this_iteration",
        ls_max_itr: Optional[int] = 100,
        ls_convergence_threshold: Optional[float] = 0.01,
    ):
        self.cost_function = cost_function
        self.max_itr = max_itr
        self.x_range = x_range
        self.local_best_option = local_best_option
        self.global_best_option = global_best_option
        self.ls_max_itr = ls_max_itr
        self.ls_convergence_threshold = ls_convergence_threshold
        self.particle_container = []

    def add_particle(self, particle: BaseParticle):
        particle.set_x_range(self.x_range)
        particle.set_cost_function(self.cost_function)
        particle.set_local_best_option(self.local_best_option)
        particle.initialize_particle_internals()
        self.particle_container.append(particle)

    def run_algorithm(self, x_initial: Optional[np.array] = None):
        # Set the x_initial
        if x_initial is None:
            x_initial = [
                random.uniform(self.x_range[i][0], self.x_range[i][1])
                for i in range(len(self.x_range))
            ]

        for particle in self.particle_container:
            particle.set_best_position(x_initial)

        overall_best_position = x_initial
        overall_best_cost = float("inf")

        position_history = []
        cost_history = []

        progress_bar = tqdm(total=self.max_itr, desc="Iterations")

        for _ in range(self.max_itr):
            best_positions_in_this_iteration, best_costs_in_this_iteration = [], []

            for particle in self.particle_container:
                best_x, best_cost = particle.perform_local_search(
                    max_itr=self.ls_max_itr,
                    convergence_threshold=self.ls_convergence_threshold,
                )

                best_positions_in_this_iteration.append(best_x)
                best_costs_in_this_iteration.append(best_cost)

            # Find the best solution of this iteration
            best_cost_index_in_this_iteration = np.argmin(best_costs_in_this_iteration)
            best_cost_in_this_iteration = best_costs_in_this_iteration[
                best_cost_index_in_this_iteration
            ]
            best_position_in_this_iteration = best_positions_in_this_iteration[
                best_cost_index_in_this_iteration
            ]
            if best_cost_in_this_iteration < overall_best_cost:
                overall_best_cost = best_cost_in_this_iteration
                overall_best_position = best_position_in_this_iteration

            # Find the global best particle (for use in the velocity vector):
            if self.global_best_option == "this_iteration":
                global_best_position = best_position_in_this_iteration
            elif self.global_best_option == "so_far":
                global_best_position = overall_best_position

            # Update every particle (using regularization hyper-parameters)
            for particle in self.particle_container:
                particle.update(global_best_position)

            position_history.append(overall_best_position)
            cost_history.append(overall_best_cost)

            # Update the tqdm progress bar
            progress_bar.update(1)

        progress_bar.close()

        particles = [
            {"position_history": particle.position_history}
            for particle in self.particle_container
        ]
        return (
            overall_best_position,
            overall_best_cost,
            position_history,
            cost_history,
            particles,
        )
