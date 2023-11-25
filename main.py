import yaml
from typing import Dict
from utils import search_algorithms, cost_functions, plot_utils, particle_class, neighborhood_generation

def main(config: Dict) -> None:
    # Get the cost function
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
        # generate neighborhoods based on x-range
        neighborhoods = neighborhood_generation.generate_neighborhoods(5, 5, x_range)

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

        print("Top Neighborhoods =>\n")
        for neighborhood in top_neighborhoods:
            # rounding the float values
            neighborhood['neighborhood'] = [ [round(i, 2) for i in elem] for elem in neighborhood['neighborhood'] ]
            neighborhood['best_x'] = [ round(elem, 2) for elem in neighborhood['best_x'] ]
            print("Neighborhood range: ", neighborhood['neighborhood'])
            print("Neighborhood cost: %.2f" % neighborhood['best_cost'])
            print("Location in neighborhood: ", neighborhood['best_x'])
            
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
                    x_range=x_range,
                )

            if config["cost_function"] == "revenue":
                print(f"Cost of trucks: {neighborhood['best_x'][0]}")
                print(f"Cost of sedans: {neighborhood['best_x'][1]}")

if __name__ == "__main__":
    with open("./config/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    main(config=config)
