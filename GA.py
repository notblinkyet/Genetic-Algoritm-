# Импортируем модули

import numpy as np
import matplotlib.pyplot as plt

# Создаем одну особь случайной перестановкой городов


def make_individ(cities: np.ndarray[float]) -> np.ndarray[float]:
    individ = np.random.permutation(cities)
    return individ

# Создаем популяцию особей


def make_population(cities: np.ndarray[float], num_population: int) -> np.ndarray[np.ndarray[float]]:
    population = np.zeros((num_population, cities.shape[0], cities.shape[1]), dtype=float)
    for i in range(num_population):
        population[i] = make_individ(cities=cities)
    return population

# Приспособленность особей


def make_fitness(population: np.ndarray[np.ndarray[float]]) -> np.ndarray[np.ndarray[float]]:
    fitness = np.zeros(len(population))
    for i in range(len(population)):
        distanse = 0
        cur = population[i]
        for j in range(1, len(cur)):
            distanse += np.sqrt((cur[j-1, 0] - cur[j, 0]) ** 2 + (cur[j-1, 1] - cur[j, 1])**2)
        distanse += np.sqrt((cur[0, 0] - cur[-1, 0]) ** 2 + (cur[0, 1] - cur[-1, 1])**2)
        fitness[i] = -1 * distanse
    return fitness
# Выбор лучших особей для создания потомства


def selection(population: np.ndarray[np.ndarray[float]], fitness: np.ndarray[np.ndarray[float]], num_parents: int) -> \
        np.ndarray[np.ndarray[int]]:
    return population[np.argsort(fitness)[:num_parents]]


# Выбираем худших для мутации
def select_worst(population: np.ndarray[np.ndarray[float]], fitness: np.ndarray[np.ndarray[float]], num_parents: int) -> \
        np.ndarray[np.ndarray[int]]:
    return population[np.argsort(fitness)[num_parents:]]

# Скрещевание двух особей


def crossing_two(first: np.ndarray[float], second: np.ndarray[float], num_cities: int) -> np.ndarray[float]:
    child = np.zeros_like(first)

    cross_point = np.random.randint(1, num_cities - 1)

    child[:cross_point] = first[:cross_point]

    child[cross_point:] = [city for city in second if city not in first[:cross_point]]

    return child

# Скрещевание популяции


def crossing(parents: np.ndarray, num_cities: int, num_children: int) -> np.ndarray[np.ndarray[float]]:
    children = np.zeros_like(parents)
    for i in range(num_children):
        first, second = np.random.choice(len(parents), size=2, replace=False)
        parent1, parent2 = parents[first], parents[second]
        children[i] = crossing_two(parent1, parent2, num_cities)
    return children

# Мутация


def mutation(worst: np.ndarray[np.ndarray[float]], threshold: float) -> np.ndarray[np.ndarray[float]]:
    for i in range(len(worst)):
        rand = np.random.rand()
        if rand < threshold:
            mutation_point1, mutation_point2 = np.random.choice(len(worst[i]), size=2, replace=False)
            if mutation_point1 > mutation_point2:
                mutation_point1, mutation_point2 = mutation_point2, mutation_point1
            worst[i, mutation_point1:mutation_point2] = worst[i, mutation_point1:mutation_point2][::-1]
        elif rand < 0.75:
            worst[i] = np.random.permutation(worst[i])
    return worst

# Главная функция программы


def main(dots: np.ndarray[float], population_size: int, generations: int, threshold: float):
    res = []
    num_cities = len(dots)
    population = make_population(dots, population_size)

    for generation in range(generations):
        fitness1 = make_fitness(population)
        parents = selection(population, fitness1, population_size // 2)
        children = crossing(parents, num_cities, population_size - len(parents))
        population = np.concatenate((parents, children))
        fitness2 = make_fitness(population)
        worst = select_worst(population, fitness2, population_size // 2)
        elit = population[population_size // 2:]
        new = mutation(worst, threshold)
        population = np.concatenate((elit, new))
        fitness3 = make_fitness(population)
        best_road_pop = population[np.argmax(fitness3)]
        best_distance_pop = -fitness3.max()
        print(f"Generation {generation + 1}, Best Distance: {best_distance_pop}")
        res.append((best_road_pop, best_distance_pop))

    return min(res, key=lambda x: x[1])

# Создание случайных точек
if __name__ == "__main__":
    # dots = np.random.random(14).reshape(7, 2)
    dots = np.array([0, 0.4, 0.1, 0.3, 0.2, 0.2, 0.3, 0.1, 0.4, 0, 0.5, 0, 0.6, 0.1, 0.7, 0.2, 0.8, 0.3, 0.9, 0.4]).reshape(10, 2)
    dots += np.random.uniform(0.01, 0.1, (10, 2))
    res = main(dots, 2000, 50, 0.4)

    # Результат
    best_road, best_distance = res
    print(best_distance)
    x_values, y_values = best_road[:, 0], best_road[:, 1]
    plt.plot(x_values, y_values, marker='o', linestyle='-')
    plt.title('График точек')
    plt.xlabel('X-координата')
    plt.ylabel('Y-координата')
    plt.show()