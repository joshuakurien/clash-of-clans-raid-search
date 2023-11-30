import yaml
from typing import Dict
from utils import search_algorithms, cost_functions, plot_utils, particle_class, neighborhood_generation
import json 
import os
import time

def main(config: Dict) -> None:
    # Get the cost function
    best_cost = None
    runtime_pso = None
    runtime_ls = None
    runtime_ga = None
    runtime_pso_regular = None
    
    if config["cost_function"] == "sphere":
        cost_function = cost_functions.sphere
        x_range = [
            [-100, 100] for i in range(config["dimension"])
        ]  # The range for each dimension
    elif config["cost_function"] == "schwefel":
        cost_function = cost_functions.schwefel
        x_range = [
            [-500, 500] for i in range(config["dimension"])
        ]  # The range for each dimension
    elif config["cost_function"] == "schaffer":
        cost_function = cost_functions.schaffer
        x_range = [
            [-100, 100] for i in range(config["dimension"])
        ]  # The range for each dimension
    elif config["cost_function"] == "griewank":
        cost_function = cost_functions.griewank
        x_range = [
            [-100, 100] for i in range(config["dimension"])
        ]  # The range for each dimension
    elif config["cost_function"] == "func7":
        cost_function = cost_functions.func7
        x_range = [
            [-1000, 1000] for i in range(config["dimension"])
        ]  # The range for each dimension
    elif config["cost_function"] == "func8":
        cost_function = cost_functions.func8
        x_range = [
            [-32, 32] for i in range(config["dimension"])
        ]  # The range for each dimension
    elif config["cost_function"] == "func9":
        cost_function = cost_functions.func9
        x_range = [
            [-5, 5] for i in range(config["dimension"])
        ]  # The range for each dimension
    elif config["cost_function"] == "func11":
        cost_function = cost_functions.func11
        x_range = [
            [-0.5, 0.5] for i in range(config["dimension"])
        ]  # The range for each dimension
    elif config["cost_function"] == "revenue":
        cost_function = cost_functions.revenue
        x_range = [
            [0, 20000] for i in range(config["dimension"])
        ]  # The range for each dimension (it must be 2 for this function)

    # Get the search algorithm
    if config["search_algorithm"] == "pso":
        start_time_pso = time.time()
        # generate neighborhoods based on x-range
        neighborhoods = neighborhood_generation.generate_neighborhoods(4, 4, x_range)

        # Set up PSO Container that runs algorithm for each particle
        pso_container = particle_class.PSO(
            cost_function=cost_function,
            max_itr=config["pso"]["max_itr"],
            x_range=x_range,
            local_best_option=config["pso"]["local_best_option"],
            global_best_option=config["pso"]["global_best_option"],
            ls_max_itr=config["pso"]["local_search"]["max_itr"],
            ls_convergence_threshold=config["pso"]["local_search"][
                "convergence_threshold"
            ],
            neighborhoods=neighborhoods
        )
        
        elixir_max = config["pso"]["max_elixir"]
        troop_types = config["pso"]["troops"]

        troops = particle_class.ClashParticle(elixir_max,troop_types)
        for troop in troops.troop_list:
            pso_container.add_particle(troop)

        # Run PSO algorithm with added particles
        top_neighborhoods = pso_container.run_algorithm_across_neighborhoods()

        # reset neighborhoods in pso container
        top_neighborhoods_list = []
        print("Top Neighborhoods before re-iterations =>\n")
        for neighborhood in top_neighborhoods:
            top_neighborhoods_list.append(neighborhood['neighborhood'])
            neighborhood_generation.print_neighborhood(neighborhood)

        pso_container.neighborhoods = top_neighborhoods_list
        
        # reiterate over neighborhoods to find better costs
        cost_improved = True
        while(cost_improved):
            top_neighborhoods_iter = pso_container.run_algorithm_across_neighborhoods()
            neighborhood_replaced = False
            for i in range(len(top_neighborhoods_iter)):
                if (top_neighborhoods_iter[i]['best_cost'] < top_neighborhoods[i]['best_cost']):
                    top_neighborhoods[i] = top_neighborhoods_iter[i]
                    neighborhood_replaced = True
            cost_improved = neighborhood_replaced

        print("Top Neighborhoods after re-iterations =>\n")
        if (config["neighborhood_plot_method"] == "all"):
            plot_utils.plot_results_multiple_neighborhoods(
                cost_function=cost_function,
                x_range=x_range,
                neighborhood_list=top_neighborhoods
            )
        elif (config["neighborhood_plot_method"] == "single"):
            for neighborhood in top_neighborhoods:
                neighborhood_generation.print_neighborhood(neighborhood)
                
                # plotting the neighborhood
                if len(neighborhood['best_x']) == 2:
                # If the dimensionality is 2, visualize the results.
                    plot_utils.plot_results(
                    best_x=neighborhood['best_x'],
                    best_cost=neighborhood['best_cost'],
                    x_history=neighborhood['x_history'],
                    cost_history=neighborhood['cost_history'],
                    cost_function=cost_function,
                    x_range=neighborhood['neighborhood'],
                )
                if (config["search_algorithm"] == "pso") or (
                    config["search_algorithm"] == "ga"
                ):
                    plot_utils.plot_results_with_population(
                        best_x=neighborhood['best_x'],
                        individuals=neighborhood['individuals'],
                        cost_function=cost_function,
                        x_range=neighborhood['neighborhood'],
                    )
                else: 
                    for neighborhood in top_neighborhoods:
                        neighborhood_generation.print_neighborhood(neighborhood)

                    best_neighborood = min(top_neighborhoods, key=lambda x: x['best_cost'])
                    print ("Best neighborhood =>")
                    neighborhood_generation.print_neighborhood(best_neighborood)
                    
        final_best_costs = [neighborhood['best_cost'] for neighborhood in top_neighborhoods]
        best_cost = min(final_best_costs)
        end_time_pso = time.time()
        runtime_pso = end_time_pso - start_time_pso



    elif config['search_algorithm'] == 'local_search':
        start_time_ls = time.time()
        best_x, best_cost, x_history, cost_history = search_algorithms.local_search(cost_function=cost_function, max_itr=config['local_search']['max_itr'],
                                                                                    convergence_threshold=config['local_search']['convergence_threshold'],
                                                                                    x_initial=config['x_initial'], x_range=x_range)
        end_time_ls = time.time()
        runtime_ls = end_time_ls - start_time_ls

        plot_utils.plot_results(best_x=best_x, best_cost=best_cost,
                                x_history=x_history, cost_history=cost_history,
                                cost_function=cost_function, x_range=x_range)
    elif config['search_algorithm'] == 'ga':
        start_time_ga = time.time()
        best_x, best_cost, x_history, cost_history, individuals = search_algorithms.ga(cost_function=cost_function, population_size=config['ga']['population_size'], max_itr=config['ga']['max_itr'],
                                                                                       mutation_rate=config['ga']['mutation_rate'], crossover_rate=config['ga']['crossover_rate'], x_initial=config['x_initial'],
                                                                                       x_range=x_range)
        end_time_ga = time.time()
        runtime_ga = end_time_ga - start_time_ga

        plot_utils.plot_results_with_population(best_x=best_x, individuals=individuals,
                                                    cost_function=cost_function, x_range=x_range)
    elif config['search_algorithm'] == 'pso_regular':
        start_time_pso_regular = time.time()
        best_x, best_cost, x_history, cost_history, individuals = search_algorithms.pso_regular(cost_function=cost_function, num_particles=config['pso_regular']['num_particles'], max_itr=config['pso_regular']['max_itr'],
                                                                                        alpha_1=config['pso_regular']['alpha_1'], alpha_2=config['pso_regular']['alpha_2'], alpha_3=config['pso_regular']['alpha_3'],
                                                                                        x_initial=config['x_initial'], x_range=x_range,
                                                                                        local_best_option=config['pso_regular']['local_best_option'],
                                                                                        global_best_option=config['pso_regular']['global_best_option'],
                                                                                        ls_max_itr=config['pso_regular']['local_search']['max_itr'], ls_convergence_threshold=config['pso_regular']['local_search']['convergence_threshold'])
        end_time_pso_regular = time.time()
        runtime_pso_regular = end_time_pso_regular - start_time_pso_regular
        plot_utils.plot_results_with_population(best_x=best_x, individuals=individuals,
                                                    cost_function=cost_function, x_range=x_range)

    results = {
        'algorithm': config['search_algorithm'],
        'cost_function': config['cost_function'],
        'final_cost': best_cost,
        'runtime': runtime_pso if config['search_algorithm'] == 'pso' else runtime_ls if config['search_algorithm'] == 'local_search' else runtime_ga if config['search_algorithm'] == 'ga' else runtime_pso_regular
    }
    
    if os.path.exists('results.json'):
        with open('results.json', 'r') as f:
            try:
                existing_results = json.load(f)
            except json.JSONDecodeError:
                existing_results = []
    else:
        existing_results = []

    existing_results.append(results)
    with open('results.json', 'w') as f:
        json.dump(existing_results, f, indent=4)

if __name__ == "__main__":
    with open("./config/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    main(config=config)
