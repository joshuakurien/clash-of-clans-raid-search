import yaml
from typing import Dict
from utils import search_algorithms, cost_functions, plot_utils


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
        # Set up PSO Container that runs algorithm for each particle
        pso_container = search_algorithms.PSO(
            cost_function=cost_function,
            max_itr=config["pso"]["max_itr"],
            x_range=x_range,
            local_best_option=config["pso"]["local_best_option"],
            global_best_option=config["pso"]["global_best_option"],
            ls_max_itr=config["pso"]["local_search"]["max_itr"],
            ls_convergence_threshold=config["pso"]["local_search"][
                "convergence_threshold"
            ],
        )

        # Add the particles required for PSO
        for particle_i in range(config["pso"]["num_particles"]):
            particle = search_algorithms.DefaultParticle(
                alpha_1=config["pso"]["alpha_1"],
                alpha_2=config["pso"]["alpha_2"],
                alpha_3=config["pso"]["alpha_3"],
            )
            pso_container.add_particle(particle)

        # Run PSO algorithm with added particles
        (
            best_x,
            best_cost,
            x_history,
            cost_history,
            individuals,
        ) = pso_container.run_algorithm(x_initial=config["x_initial"])

    if len(best_x) == 2:
        # If the dimensionality is 2, visualize the results.
        plot_utils.plot_results(
            best_x=best_x,
            best_cost=best_cost,
            x_history=x_history,
            cost_history=cost_history,
            cost_function=cost_function,
            x_range=x_range,
        )
        if (config["search_algorithm"] == "pso") or (
            config["search_algorithm"] == "ga"
        ):
            plot_utils.plot_results_with_population(
                best_x=best_x,
                individuals=individuals,
                cost_function=cost_function,
                x_range=x_range,
            )

        if config["cost_function"] == "revenue":
            print(f"Cost of trucks: {best_x[0]}")
            print(f"Cost of sedans: {best_x[1]}")


if __name__ == "__main__":
    with open("./config/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    main(config=config)
