# ECE 457A Final Project - Clash of Clans Particle Swarm Optimizer
[Clash of Clans Optimizer Paper](https://github.com/joshuakurien/clash-of-clans-raid-search/files/13520445/ECE457A_ClashofClans_Algorithm.pdf)

**Authors:** Avery Chiu, Brandon Goh, Jessica Wormald, Mathurah Ravigulan, Mansheel Chahal, Joshua Kurien
![Alt text](image.png)

### Summary 
This project puts a twist on the Particle Swarm Optimization Algorithm tailored to the dynamics of the popular video game Clash of Clans. The proposed Clash of Clans Optimizer utilizes diverse troop types with unique characteristics, presenting a dynamic balance between exploration and exploitation in multi-dimensional search applications. 

## Introduction

This repository contains the implementation of a novel metaheuristic optimization method, the Clash of Clans Optimizer. The algorithm is inspired by Particle Swarm Optimization (PSO) and tailored to the dynamics of the popular video game Clash of Clans.

## Features

- Diverse troop types with unique characteristics
- Dynamic balance between exploration and exploitation
- Tunability through elixir amounts
- Introduction of hard-coding troop compositions
- Integration of Variable Neighborhood Search approach

## Getting Started

### Prerequisites

- Python 3.7+

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/joshuakurien/clash-of-clans-raid-search
   cd clash-of-clans-raid-search
   ```

2. Install dependencies
    ```
    pip install -r requirements.txt
    ```
    
### Instructions to run
Run the `main.py` file. Edit the `config.py` to customize which algorithm to run and cost function to run. Available options: 
- `pso`: Clash of Clans PSO algorithm 
- `pso_regular`: Regular PSO algorithm
- `ga`: Genetic Algorithm
- `local_search`: Local Search


## Baseline

| Algorithm               | Cost Function | Runtime (s) | Final cost               |
|-------------------------|---------------|-------------|--------------------------|
| Clash of Clans Optimizer | Revenue       | 27.8        | -181499999.99999952     |
|                         | Griewank      | 18.7        | 0                        |
|                         | Schwefel      | 16.48       | 6.689                    |
|                         | Schaffer      | 22.98       | 0                        |
| PSO                     | Revenue       | 1.01        | -181377617.65602878     |
|                         | Griewank      | 0.0478      | 0.0073960944278980145   |
|                         | Schwefel      | 2.83        | 123.70750017327083      |
|                         | Schaffer      | 1.828       | 0.0159                   |
| Genetic Algorithm       | Revenue       | 0.027       | -181457732.8459237      |
|                         | Griewank      | 0.0478      | 0.017668157859585754    |
|                         | Schwefel      | 0.0497      | 5.167                    |
|                         | Schaffer      | 0.0396      | 2.5148                   |
| Local Search             | Revenue       | 0.011       | 274225900.64544237      |
|                         | Griewank      | 0.008       | 0.7173885164256175      |
|                         | Schwefel      | 0.00899     | 510.6                    |
|                         | Schaffer      | 0.008998    | 10.01                    |

## Acknowledgment

We would like to express our sincere appreciation to Professor Benyamin Ghojogh for his exceptional guidance and inspirational teaching in the realm of metaheuristic optimization methods. Professor Ghojogh's passion for the subject, coupled with his clarity in conveying complex mathematical concepts, has not only deepened our understanding but has also ignited a genuine enthusiasm for exploring the frontiers of this fascinating field. His unwavering commitment to academic excellence and his ability to inspire curiosity have left an indelible mark on our learning journey. We are grateful for the invaluable mentorship and inspiration provided by Professor Ghojogh, which will undoubtedly shape our future pursuits in this area. We acknowledge Professor Benyamin Ghojogh for being an inspiring educator and mentor.

## References

1. S. Darvishpoor, A. Darvishpour, M. Escarcega, and M. Hassanalian, “Nature-inspired algorithms from oceans to space: A comprehensive review of heuristic and meta-heuristic optimization algorithms and their potential applications in drones,” *Drones*, vol. 7, no. 7, p. 427, 2023.

2. J. Kennedy and R. Eberhart, “Particle swarm optimization,” in *Proceedings of ICNN’95-international conference on neural networks*, vol. 4, pp. 1942–1948, IEEE, 1995.

3. Clash of Clans Wiki. (n.d.). [Online]. Available: [Clash of Clans Wiki](https://clashofclans.fandom.com/wiki).


