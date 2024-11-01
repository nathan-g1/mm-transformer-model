import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI
import concurrent.futures
import time

# Generate dummy image data (grayscale values)
def generate_image_data(size):
    return np.random.randint(0, 256, (size, size))

# Basic K-means function
def kmeans(data, k, max_iters=100):
    centroids = data[np.random.choice(data.shape[0], k, replace=False)]
    for _ in range(max_iters):
        distances = np.sqrt(((data - centroids[:, np.newaxis])**2).sum(axis=2))
        clusters = np.argmin(distances, axis=0)
        new_centroids = np.array([data[clusters == i].mean(axis=0) for i in range(k)])
        if np.all(centroids == new_centroids):
            break
        centroids = new_centroids
    return centroids

# Sequential K-means clustering
def kmeans_sequential(data, k, max_iters=100):
    return kmeans(data, k, max_iters)

# K-means with threading
def kmeans_with_threads(data, k, max_iters=100, n_threads=4):
    data_split = np.array_split(data, n_threads)
    centroids = np.random.rand(k, data.shape[1])
    
    def process_segment(segment):
        return kmeans(segment, k, max_iters)
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_threads) as executor:
        results = executor.map(process_segment, data_split)
    
    centroids = np.mean([res for res in results], axis=0)
    return centroids

# K-means with multiprocessing (MPI-like simulation)
def kmeans_with_mpi(data, k, max_iters=100, n_processes=4):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    data_split = np.array_split(data, size) if rank == 0 else None
    data_segment = comm.scatter(data_split, root=0)

    local_centroids = kmeans(data_segment, k, max_iters)
    centroids = comm.gather(local_centroids, root=0)

    if rank == 0:
        centroids = np.mean(centroids, axis=0)
    return centroids

# Timing function
def time_algorithm(data, k, func, *args):
    start_time = time.time()
    func(data, k, *args)
    return time.time() - start_time

# Execution time for each algorithm with different `k` values
def compare_execution_time_k(data, k_values):
    seq_times = []
    thread_times = []
    mpi_times = []
    
    for k in k_values:
        seq_times.append(time_algorithm(data, k, kmeans_sequential))
        thread_times.append(time_algorithm(data, k, kmeans_with_threads, 4))
        mpi_times.append(time_algorithm(data, k, kmeans_with_mpi, 4))
    
    plt.plot(k_values, seq_times, label='Sequential')
    plt.plot(k_values, thread_times, label='Threaded')
    plt.plot(k_values, mpi_times, label='MPI')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Execution Time (s)')
    plt.legend()
    plt.show()

# Speedup comparison between algorithms
def compare_speedup(data, k, max_threads=8):
    sequential_time = time_algorithm(data, k, kmeans_sequential)

    thread_speedups = []
    mpi_speedups = []

    for n in range(1, max_threads + 1):
        thread_time = time_algorithm(data, k, kmeans_with_threads, n)
        mpi_time = time_algorithm(data, k, kmeans_with_mpi, n)

        # Calculate speedup as (sequential time / parallel time)
        thread_speedups.append(sequential_time / thread_time)
        mpi_speedups.append(sequential_time / mpi_time)

    # Plotting speedup for each approach
    plt.plot(range(1, max_threads + 1), thread_speedups, label='Threaded Speedup', marker='o')
    plt.plot(range(1, max_threads + 1), mpi_speedups, label='MPI Speedup', marker='x')
    plt.xlabel('Number of Threads/Processes')
    plt.ylabel('Speedup')
    plt.legend()
    plt.title('Speedup Comparison Across Threads/Processes')
    plt.show()

# Execution time with different image sizes
def compare_execution_time_size(sizes, k):
    seq_times = []
    thread_times = []
    mpi_times = []
    
    for size in sizes:
        data = generate_image_data(size)
        seq_times.append(time_algorithm(data, k, kmeans_sequential))
        thread_times.append(time_algorithm(data, k, kmeans_with_threads, 4))
        mpi_times.append(time_algorithm(data, k, kmeans_with_mpi, 4))
    
    plt.plot(sizes, seq_times, label='Sequential')
    plt.plot(sizes, thread_times, label='Threaded')
    plt.plot(sizes, mpi_times, label='MPI')
    plt.xlabel('Image Size')
    plt.ylabel('Execution Time (s)')
    plt.legend()
    plt.show()

# New function to compare execution times of threaded and MPI-based K-means with varying threads/processes
def compare_execution_time_threads_processes(data, k, max_threads=8):
    thread_times = []
    mpi_times = []

    for n in range(1, max_threads + 1):
        thread_times.append(time_algorithm(data, k, kmeans_with_threads, n))
        mpi_times.append(time_algorithm(data, k, kmeans_with_mpi, n))

    plt.plot(range(1, max_threads + 1), thread_times, label='Threaded')
    plt.plot(range(1, max_threads + 1), mpi_times, label='MPI')
    plt.xlabel('Number of Threads/Processes')
    plt.ylabel('Execution Time (s)')
    plt.legend()
    plt.show()

# Example of usage
data = generate_image_data(1000)  # A 1000x1000 image
k_values = [2, 4, 8, 16]
sizes = [100, 200, 400, 800]

# compare_execution_time_k(data, k_values)
# compare_speedup(data, k=4)
# compare_execution_time_size(sizes, k=4)
# compare_execution_time_threads_processes(data, k=4, max_threads=8)
