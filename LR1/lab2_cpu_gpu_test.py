import numpy as np
import torch
import matplotlib.pyplot as plt

def tic():
    # Homemade version of matlab tic and toc functions
    import time
    global startTime_for_tictoc
    startTime_for_tictoc = time.time()


def toc():
    import time
    if 'startTime_for_tictoc' in globals():
        print ("Elapsed time is " + str(time.time() - startTime_for_tictoc) + " seconds.")
        return time.time() - startTime_for_tictoc
    else:
        print ("Toc: start time not set")

def mat_generator(size_x, size_y, int_range=10):
    array = np.random.randint(int_range, size=(size_y, size_x))
    # print(array)
    return array


def mat_copy_gpu(matrix):
    tensor1 = torch.from_numpy(matrix)
    return tensor1


def plotter(cpu_timings, gpu_timings, iter_list, label_x, label_y):
    # fig = plt.figure()
    plt.plot(iter_list, cpu_timings, label='CPU')
    plt.legend()
    plt.plot(iter_list, gpu_timings, label='GPU')
    plt.legend()
    plt.grid()
    plt.xlabel(label_x)
    plt.ylabel(label_y)

    plt.savefig("figure.png")
    plt.show()

def vector_sum_cpu(A):
    count = 0
    tic()
    for i in range(len(A)):
        count = count + A[i]
    running_time = toc()
    return running_time


def vector_sum_gpu(A):
    tensor = mat_copy_gpu(A)
    tic()
    count = torch.sum(tensor)
    running_time = toc()

    return running_time


if __name__ == '__main__':
    iterations = 200
    step = 5000
    max_size = step * iterations
    min_size = 1000
    SIZE = min_size
    label_x = "Количество суммируемых элементов"
    label_y = "Время суммирования элементов, сек"

    cpu_timings = []
    gpu_timings = []
    iter_list = []

    while SIZE <= max_size:
        X = mat_generator(1, SIZE)
        cpu_timings.append(vector_sum_cpu(X))
        gpu_timings.append(vector_sum_gpu(X))
        iter_list.append(SIZE)
        SIZE = SIZE + step
    plotter(cpu_timings, gpu_timings, iter_list, label_x, label_y)
    print("done")