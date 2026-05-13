#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>
#include <mpi.h>
#include <cuda_runtime.h>

#define N 4096
#define SOURCE 1
#define MAXINT 9999999
#define RUNS 1
#define BLOCK_SIZE 256

#define CHECK_CUDA(call) do {                                      \
    cudaError_t err = call;                                        \
    if (err != cudaSuccess) {                                      \
        printf("CUDA error at %s:%d: %s\n",                        \
               __FILE__, __LINE__, cudaGetErrorString(err));       \
        MPI_Abort(MPI_COMM_WORLD, 1);                              \
    }                                                             \
} while (0)

static double now_seconds(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec + (double)tv.tv_usec / 1000000.0;
}

static void fill_random_graph(int *graph, int density_pct, unsigned int seed) {
    srand(seed);

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (i == j) {
                graph[i * N + j] = 0;
            } else if ((rand() % 100) < density_pct) {
                graph[i * N + j] = (rand() % 100) + 1;
            } else {
                graph[i * N + j] = MAXINT;
            }
        }
    }
}

static void serial_dijkstra(const int *graph, int source, int *dist) {
    unsigned char *visited = (unsigned char *)malloc(N * sizeof(unsigned char));
    if (!visited) {
        printf("serial_dijkstra malloc failed\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    for (int i = 0; i < N; i++) {
        dist[i] = graph[source * N + i];
        visited[i] = 0;
    }

    dist[source] = 0;
    visited[source] = 1;

    for (int count = 1; count < N; count++) {
        int minDist = MAXINT;
        int u = -1;

        for (int i = 0; i < N; i++) {
            if (!visited[i] && dist[i] < minDist) {
                minDist = dist[i];
                u = i;
            }
        }

        if (u == -1) break;

        visited[u] = 1;

        for (int v = 0; v < N; v++) {
            int w = graph[u * N + v];

            if (!visited[v] && w < MAXINT && minDist + w < dist[v]) {
                dist[v] = minDist + w;
            }
        }
    }

    free(visited);
}

static void serial_dijkstra_apsp(const int *graph, int *distAll) {
    for (int source = 0; source < N; source++) {
        serial_dijkstra(graph, source, &distAll[source * N]);
    }
}

static void prepare_mpi_sendbuf(const int *graph, int *sendbuf, int npes, int nlocal) {
    for (int p = 0; p < npes; p++) {
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < nlocal; j++) {
                sendbuf[p * N * nlocal + i * nlocal + j] =
                    graph[i * N + p * nlocal + j];
            }
        }
    }
}

static void mpi_dijkstra(int n,
                         int source,
                         int *localWeight,
                         int *localDistance,
                         MPI_Comm comm) {
    int npes, myrank;
    MPI_Comm_size(comm, &npes);
    MPI_Comm_rank(comm, &myrank);

    int nlocal = n / npes;
    int firstvtx = myrank * nlocal;

    int *marker = (int *)malloc(nlocal * sizeof(int));
    if (!marker) {
        printf("Rank %d marker malloc failed\n", myrank);
        MPI_Abort(comm, 1);
    }

    for (int j = 0; j < nlocal; j++) {
        localDistance[j] = localWeight[source * nlocal + j];
        marker[j] = 1;
    }

    if (source >= firstvtx && source < firstvtx + nlocal) {
        marker[source - firstvtx] = 0;
        localDistance[source - firstvtx] = 0;
    }

    for (int i = 1; i < n; i++) {
        int lminpair[2] = {MAXINT, -1};
        int gminpair[2] = {MAXINT, -1};

        for (int j = 0; j < nlocal; j++) {
            if (marker[j] && localDistance[j] < lminpair[0]) {
                lminpair[0] = localDistance[j];
                lminpair[1] = firstvtx + j;
            }
        }

        MPI_Allreduce(lminpair, gminpair, 1, MPI_2INT, MPI_MINLOC, comm);

        int udist = gminpair[0];
        int u = gminpair[1];

        if (u < 0 || udist >= MAXINT) break;

        if (u >= firstvtx && u < firstvtx + nlocal) {
            marker[u - firstvtx] = 0;
        }

        for (int j = 0; j < nlocal; j++) {
            int w = localWeight[u * nlocal + j];
            if (marker[j] && w < MAXINT && udist + w < localDistance[j]) {
                localDistance[j] = udist + w;
            }
        }
    }

    free(marker);
}

__global__ void init_kernel(const int *graph, const int *sourcePtr, int *dist, unsigned char *visited) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N) {
        int source = *sourcePtr;
        dist[i] = graph[source * N + i];
        visited[i] = 0;

        if (i == source) {
            dist[i] = 0;
            visited[i] = 1;
        }
    }
}

__global__ void find_block_min_kernel(const int *dist,
                                      const unsigned char *visited,
                                      int *blockMinDist,
                                      int *blockMinIndex) {
    __shared__ int sDist[BLOCK_SIZE];
    __shared__ int sIndex[BLOCK_SIZE];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;

    if (i < N && !visited[i]) {
        sDist[tid] = dist[i];
        sIndex[tid] = i;
    } else {
        sDist[tid] = MAXINT;
        sIndex[tid] = -1;
    }

    __syncthreads();

    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            if (sDist[tid + stride] < sDist[tid]) {
                sDist[tid] = sDist[tid + stride];
                sIndex[tid] = sIndex[tid + stride];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        blockMinDist[blockIdx.x] = sDist[0];
        blockMinIndex[blockIdx.x] = sIndex[0];
    }
}

__global__ void find_global_min_kernel(const int *blockMinDist,
                                       const int *blockMinIndex,
                                       int *globalMinDist,
                                       int *globalMinIndex,
                                       int numBlocks) {
    __shared__ int sDist[BLOCK_SIZE];
    __shared__ int sIndex[BLOCK_SIZE];

    int tid = threadIdx.x;

    if (tid < numBlocks) {
        sDist[tid] = blockMinDist[tid];
        sIndex[tid] = blockMinIndex[tid];
    } else {
        sDist[tid] = MAXINT;
        sIndex[tid] = -1;
    }

    __syncthreads();

    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            if (sDist[tid + stride] < sDist[tid]) {
                sDist[tid] = sDist[tid + stride];
                sIndex[tid] = sIndex[tid + stride];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        *globalMinDist = sDist[0];
        *globalMinIndex = sIndex[0];
    }
}

__global__ void update_kernel(const int *graph,
                              int *dist,
                              unsigned char *visited,
                              const int *globalMinDist,
                              const int *globalMinIndex) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= N) return;

    int u = *globalMinIndex;
    int uDist = *globalMinDist;

    if (u < 0 || uDist >= MAXINT) return;

    if (v == u) {
        visited[v] = 1;
        return;
    }

    if (visited[v]) return;

    int w = graph[u * N + v];
    if (w < MAXINT && uDist + w < dist[v]) {
        dist[v] = uDist + w;
    }
}

static void enqueue_cuda_dijkstra(cudaStream_t stream,
                                  int *d_graph,
                                  int *d_source,
                                  int *d_dist,
                                  unsigned char *d_visited,
                                  int *d_blockMinDist,
                                  int *d_blockMinIndex,
                                  int *d_globalMinDist,
                                  int *d_globalMinIndex) {
    int numBlocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    init_kernel<<<numBlocks, BLOCK_SIZE, 0, stream>>>(d_graph, d_source, d_dist, d_visited);

    for (int i = 1; i < N; i++) {
        find_block_min_kernel<<<numBlocks, BLOCK_SIZE, 0, stream>>>(
            d_dist,
            d_visited,
            d_blockMinDist,
            d_blockMinIndex
        );

        find_global_min_kernel<<<1, BLOCK_SIZE, 0, stream>>>(
            d_blockMinDist,
            d_blockMinIndex,
            d_globalMinDist,
            d_globalMinIndex,
            numBlocks
        );

        update_kernel<<<numBlocks, BLOCK_SIZE, 0, stream>>>(
            d_graph,
            d_dist,
            d_visited,
            d_globalMinDist,
            d_globalMinIndex
        );
    }
}

static double run_cuda_graph_dijkstra_apsp(const int *hostGraph, int *cudaDistAll) {
    int numBlocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    int *d_graph = NULL;
    int *d_source = NULL;
    int *d_dist = NULL;
    unsigned char *d_visited = NULL;
    int *d_blockMinDist = NULL;
    int *d_blockMinIndex = NULL;
    int *d_globalMinDist = NULL;
    int *d_globalMinIndex = NULL;

    CHECK_CUDA(cudaMalloc(&d_graph, (size_t)N * N * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_source, sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_dist, N * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_visited, N * sizeof(unsigned char)));
    CHECK_CUDA(cudaMalloc(&d_blockMinDist, numBlocks * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_blockMinIndex, numBlocks * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_globalMinDist, sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_globalMinIndex, sizeof(int)));

    CHECK_CUDA(cudaMemcpy(d_graph, hostGraph, (size_t)N * N * sizeof(int), cudaMemcpyHostToDevice));

    cudaStream_t stream;
    cudaGraph_t graphObj;
    cudaGraphExec_t graphExec;

    CHECK_CUDA(cudaStreamCreate(&stream));
    CHECK_CUDA(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));

    enqueue_cuda_dijkstra(
        stream,
        d_graph,
        d_source,
        d_dist,
        d_visited,
        d_blockMinDist,
        d_blockMinIndex,
        d_globalMinDist,
        d_globalMinIndex
    );

    CHECK_CUDA(cudaStreamEndCapture(stream, &graphObj));
    CHECK_CUDA(cudaGraphInstantiate(&graphExec, graphObj, NULL, NULL, 0));

    /* Warm-up one source so graph setup/JIT effects are not included. */
    int warmupSource = 0;
    CHECK_CUDA(cudaMemcpy(d_source, &warmupSource, sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaGraphLaunch(graphExec, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    float cudaTotalMs = 0.0f;

    for (int r = 0; r < RUNS; r++) {
        CHECK_CUDA(cudaEventRecord(start, stream));

        for (int source = 0; source < N; source++) {
            CHECK_CUDA(cudaMemcpy(d_source, &source, sizeof(int), cudaMemcpyHostToDevice));
            CHECK_CUDA(cudaGraphLaunch(graphExec, stream));
        }

        CHECK_CUDA(cudaEventRecord(stop, stream));
        CHECK_CUDA(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
        cudaTotalMs += ms;
    }

    /* Copy one full APSP result for correctness checking. */
    for (int source = 0; source < N; source++) {
        CHECK_CUDA(cudaMemcpy(d_source, &source, sizeof(int), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaGraphLaunch(graphExec, stream));
        CHECK_CUDA(cudaMemcpyAsync(&cudaDistAll[source * N], d_dist, N * sizeof(int), cudaMemcpyDeviceToHost, stream));
    }
    CHECK_CUDA(cudaStreamSynchronize(stream));

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaGraphExecDestroy(graphExec));
    CHECK_CUDA(cudaGraphDestroy(graphObj));
    CHECK_CUDA(cudaStreamDestroy(stream));

    cudaFree(d_graph);
    cudaFree(d_source);
    cudaFree(d_dist);
    cudaFree(d_visited);
    cudaFree(d_blockMinDist);
    cudaFree(d_blockMinIndex);
    cudaFree(d_globalMinDist);
    cudaFree(d_globalMinIndex);

    return (cudaTotalMs / RUNS) / 1000.0;
}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    int npes, myrank;
    MPI_Comm_size(MPI_COMM_WORLD, &npes);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    const char *mode = "all";
    if (argc >= 2) {
        mode = argv[1];
    }

    int run_serial = (strcmp(mode, "all") == 0 || strcmp(mode, "serial") == 0);
    int run_mpi = (strcmp(mode, "all") == 0 || strcmp(mode, "mpi") == 0);
    int run_cuda = (strcmp(mode, "all") == 0 || strcmp(mode, "cuda") == 0);

    if (!run_serial && !run_mpi && !run_cuda) {
        if (myrank == 0) {
            printf("Usage:\n");
            printf("  ./dijkstra_combined serial     # serial APSP only, clean baseline\n");
            printf("  mpirun -np 4 ./dijkstra_combined mpi   # MPI APSP only\n");
            printf("  ./dijkstra_combined cuda       # CUDA Graph APSP only, clean CUDA timing\n");
            printf("  ./dijkstra_combined all        # serial + MPI + CUDA, not ideal for CUDA timing\n");
        }
        MPI_Finalize();
        return 1;
    }

    if (run_cuda && strcmp(mode, "cuda") == 0 && npes != 1) {
        if (myrank == 0) {
            printf("CUDA-only mode should be run without mpirun or with -np 1.\n");
            printf("Example: ./dijkstra_combined cuda\n");
        }
        MPI_Finalize();
        return 1;
    }

    if (run_mpi && N % npes != 0) {
        if (myrank == 0) {
            printf("N must divide number of MPI processes. N=%d, processes=%d\n", N, npes);
        }
        MPI_Finalize();
        return 1;
    }

    int nlocal = run_mpi ? (N / npes) : N;
    int density = (N <= 100) ? 40 : 20;
    unsigned int seed = 42;

    int *graph = NULL;
    int *sendbuf = NULL;
    int *serialDistAll = NULL;
    int *mpiDistAll = NULL;
    int *cudaDistAll = NULL;

    int *localWeight = NULL;
    int *localDistance = NULL;

    if (run_mpi) {
        localWeight = (int *)malloc((size_t)N * nlocal * sizeof(int));
        localDistance = (int *)malloc(nlocal * sizeof(int));

        if (!localWeight || !localDistance) {
            printf("Rank %d local malloc failed\n", myrank);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    if (myrank == 0) {
        graph = (int *)malloc((size_t)N * N * sizeof(int));
        serialDistAll = (int *)malloc((size_t)N * N * sizeof(int));

        if (run_mpi) {
            sendbuf = (int *)malloc((size_t)N * N * sizeof(int));
            mpiDistAll = (int *)malloc((size_t)N * N * sizeof(int));
        }

        if (run_cuda) {
            cudaDistAll = (int *)malloc((size_t)N * N * sizeof(int));
        }

        if (!graph || !serialDistAll || (run_mpi && (!sendbuf || !mpiDistAll)) || (run_cuda && !cudaDistAll)) {
            printf("Root malloc failed\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        printf("\n=== Combined Dijkstra Benchmark ===\n");
        printf("Mode: %s\n", mode);
        printf("N: %d   Sources: all nodes (APSP)   Runs: %d\n", N, RUNS);
        printf("Density: %d%%   Seed: %u   MPI processes: %d\n", density, seed, npes);

        if (run_cuda && npes > 1) {
            printf("WARNING: CUDA is being timed inside an MPI run with %d processes.\n", npes);
            printf("For clean CUDA timing, use: ./dijkstra_combined cuda\n");
        }

        printf("\n--- Building random graph ---\n");
        fill_random_graph(graph, density, seed);

        if (run_mpi) {
            prepare_mpi_sendbuf(graph, sendbuf, npes, nlocal);
        }
    }

    double serialTotal = 0.0;
    if (myrank == 0) {
        printf("\nRunning serial Dijkstra APSP baseline...\n");
        for (int r = 0; r < RUNS; r++) {
            double t1 = now_seconds();
            serial_dijkstra_apsp(graph, serialDistAll);
            double t2 = now_seconds();
            serialTotal += t2 - t1;
        }
    }

    double mpiTotal = 0.0;
    if (run_mpi) {
        if (myrank == 0) printf("Running MPI Dijkstra APSP...\n");

        for (int r = 0; r < RUNS; r++) {
            MPI_Barrier(MPI_COMM_WORLD);
            double t1 = MPI_Wtime();

            MPI_Scatter(
                sendbuf,
                N * nlocal,
                MPI_INT,
                localWeight,
                N * nlocal,
                MPI_INT,
                0,
                MPI_COMM_WORLD
            );

            for (int source = 0; source < N; source++) {
                mpi_dijkstra(N, source, localWeight, localDistance, MPI_COMM_WORLD);

                MPI_Gather(
                    localDistance,
                    nlocal,
                    MPI_INT,
                    myrank == 0 ? &mpiDistAll[source * N] : NULL,
                    nlocal,
                    MPI_INT,
                    0,
                    MPI_COMM_WORLD
                );
            }

            MPI_Barrier(MPI_COMM_WORLD);
            double t2 = MPI_Wtime();

            if (myrank == 0) mpiTotal += t2 - t1;
        }
    }

    double cudaAvg = 0.0;
    if (run_cuda && myrank == 0) {
        printf("Running CUDA Graph Dijkstra APSP on rank 0 only...\n");
        CHECK_CUDA(cudaSetDevice(0));
        cudaAvg = run_cuda_graph_dijkstra_apsp(graph, cudaDistAll);
    }

    if (myrank == 0) {
        double serialAvg = serialTotal / RUNS;

        printf("\nAverage Results\n");
        printf("%-28s | %-14s | %-10s | %-10s\n", "Algorithm", "Time (sec)", "Speedup", "Correct");
        printf("%-28s | %14.6f | %9.3fx | %-10s\n", "Serial Dijkstra APSP", serialAvg, 1.0, "YES");

        if (run_mpi) {
            double mpiAvg = mpiTotal / RUNS;
            int mpiCorrect = 1;

            for (int idx = 0; idx < N * N; idx++) {
                if (serialDistAll[idx] != mpiDistAll[idx]) {
                    mpiCorrect = 0;
                    printf("MPI mismatch at APSP index %d: serial = %d, MPI = %d\n",
                           idx, serialDistAll[idx], mpiDistAll[idx]);
                    break;
                }
            }

            printf("%-28s | %14.6f | %9.3fx | %-10s\n", "MPI Dijkstra APSP", mpiAvg, serialAvg / mpiAvg, mpiCorrect ? "YES" : "NO");
        }

        if (run_cuda) {
            int cudaCorrect = 1;

            for (int idx = 0; idx < N * N; idx++) {
                if (serialDistAll[idx] != cudaDistAll[idx]) {
                    cudaCorrect = 0;
                    printf("CUDA mismatch at APSP index %d: serial = %d, CUDA = %d\n",
                           idx, serialDistAll[idx], cudaDistAll[idx]);
                    break;
                }
            }

            printf("%-28s | %14.6f | %9.3fx | %-10s\n", "CUDA Graph Dijkstra APSP", cudaAvg, serialAvg / cudaAvg, cudaCorrect ? "YES" : "NO");
        }
    }

    if (myrank == 0) {
        free(graph);
        free(sendbuf);
        free(serialDistAll);
        free(mpiDistAll);
        free(cudaDistAll);
    }

    free(localWeight);
    free(localDistance);

    MPI_Finalize();
    return 0;
}
