from typing import List, Optional

# divides a given range into n sets
def divide_range(start, end, n):
    step = (end - start) / n
    sets = []

    for i in range(n):
        set_start = start + i * step
        set_end = start + (i + 1) * step
        sets.append((set_start, set_end))

    return sets

# generates a neighborhood in 2d only based on the number of rows and columns 
# to divide the x_range by
def generate_neighborhoods(num_col: int, num_row: int,
                           x_range: Optional[List[List[float]]] = None):
    left_point = x_range[0][0]
    right_point = x_range[0][1]
    bottom_point = x_range[1][0]
    top_point = x_range[1][1]
    col_set = divide_range(left_point, right_point, num_col)
    row_set = divide_range(bottom_point, top_point, num_row)
    neighborhoods = []

    for i in range(num_col):
        width_neighborhood = col_set[i]
        for j in range(num_row):
            height_neighborhood = row_set[j]
            current_neighborhood = [width_neighborhood, height_neighborhood]
            neighborhoods.append(current_neighborhood)

    return neighborhoods

def print_neighborhood(neighborhood):
    # rounding the float values
    neighborhood['neighborhood'] = [ [round(i, 2) for i in elem] for elem in neighborhood['neighborhood'] ]
    neighborhood['best_x'] = [ round(elem, 2) for elem in neighborhood['best_x'] ]
    print("Neighborhood range: ", neighborhood['neighborhood'])
    print("Neighborhood cost: %.2f" % neighborhood['best_cost'])
    print("Location in neighborhood: ", neighborhood['best_x'])
    print("")