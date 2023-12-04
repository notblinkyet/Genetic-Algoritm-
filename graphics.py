import GA
import numpy as np
import time
from matplotlib import pyplot as plt


def make_dots(num_dots):
    return np.random.random(num_dots*2).reshape(num_dots, 2)


def algo(dots, generations, threshold):
    return GA.main(dots, len(dots)*100, generations, threshold)


times = np.zeros(51-3)
for i in range(3, 51):

    start = time.time()
    algo(make_dots(i), 100, 0.4)
    end = time.time()
    times[i-3] = end - start

plt.plot(np.arange(3, 51), times, marker='o')
plt.title('Зависимость времени работы от объема ввода')
plt.xlabel('Размер ввода')
plt.ylabel('Время работы (сек)')
plt.show()