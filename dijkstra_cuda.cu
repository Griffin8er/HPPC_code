#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <cuda_runtime.h>

#define INF 1000000000
#define RUNS 25
#define BLOCK_SIZE 256

#define CHECK_CUDA(call) do {                                      \
    cudaError_t err = (call);                                      \
    if (err != cudaSuccess) {                                      \
        printf("CUDA error at %s:%d: %s\n",                        \
               __FILE__, __LINE__, cudaGetErrorString(err));       \
        exit(1);                                                   \
    }                                                              \
} while (0)

typedef struct {
    int vertices;
    int **adj_matrix;
} Graph;

double now_seconds(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec + (double)tv.tv_usec / 1000000.0;
}

Graph *graph_create(int vertices) {
    Graph *g = (Graph *)malloc(sizeof(Graph));
    if (!g) {
        printf("Graph malloc failed\n");
        exit(1);
    }

    g->vertices = vertices;
    g->adj_matrix = (int **)malloc((size_t)vertices * sizeof(int *));
    if (!g->adj_matrix) {
        printf("Graph row malloc failed\n");
        exit(1);
    }

    for (int i = 0; i < vertices; i++) {
        g->adj_matrix[i] = (int *)malloc((size_t)vertices * sizeof(int));
        if (!g->adj_matrix[i]) {
            printf("Graph matrix malloc failed\n");
            exit(1);
        }

        for (int j = 0; j < vertices; j++) {
            g->adj_matrix[i][j] = (i == j) ? 0 : INF;
        }
    }

    return g;
}

void graph_add_edge(Graph *g, int src, int dest, int weight) {
    if (src != dest && g->adj_matrix[src][dest] > weight) {
        g->adj_matrix[src][dest] = weight;
    }
}

void graph_free(Graph *g) {
    if (!g) return;
    for (int i = 0; i < g->vertices; i++) {
        free(g->adj_matrix[i]);
    }
    free(g->adj_matrix);
    free(g);
}

void fill_random_edges(Graph *g, int density_pct, unsigned int seed) {
    srand(seed);
    int n = g->vertices;

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (i != j && (rand() % 100) < density_pct) {
                graph_add_edge(g, i, j, (rand() % 100) + 1);
            }
        }
    }
}

int *flatten_matrix(int **m, int n) {
    int *flat = (int *)malloc((size_t)n * n * sizeof(int));
    if (!flat) {
        printf("Flat matrix malloc failed\n");
        exit(1);
    }

    for (int i = 0; i < n; i++) {
        memcpy(flat + (size_t)i * n, m[i], (size_t)n * sizeof(int));
    }

    return flat;
}

void serial_dijkstra(const int *graph, int *dist, int n, int source) {
    unsigned char *visited = (unsigned char *)malloc((size_t)n * sizeof(unsigned char));
    if (!visited) {
        printf("Visited malloc failed\n");
        exit(1);
    }

    for (int i = 0; i < n; i++) {
        dist[i] = graph[(size_t)source * n + i];
        visited[i] = 0;
    }

    dist[source] = 0;

    for (int count = 0; count < n; count++) {
        int minDist = INF;
        int u = -1;

        for (int i = 0; i < n; i++) {
            if (!visited[i] && dist[i] < minDist) {
                minDist = dist[i];
                u = i;
            }
        }

        if (u == -1) break;

        visited[u] = 1;

        for (int v = 0; v < n; v++) {
            int w = graph[(size_t)u * n + v];

            if (!visited[v] && w < INF && minDist < INF && minDist + w < dist[v]) {
                dist[v] = minDist + w;
            }
        }
    }

    free(visited);
}

__global__ void init_kernel(
    const int *graph,
    int *dist,
    unsigned char *visited,
    int n,
    int source
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        dist[i] = graph[(size_t)source * n + i];
        visited[i] = 0;

        if (i == source) {
            dist[i] = 0;
        }
    }
}

__global__ void find_block_min_kernel(
    const int *dist,
    const unsigned char *visited,
    int *blockMinDist,
    int *blockMinIndex,
    int n
) {
    __shared__ int sDist[BLOCK_SIZE];
    __shared__ int sIndex[BLOCK_SIZE];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;

    if (i < n && !visited[i]) {
        sDist[tid] = dist[i];
        sIndex[tid] = i;
    } else {
        sDist[tid] = INF;
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

__global__ void find_global_min_kernel(
    const int *blockMinDist,
    const int *blockMinIndex,
    int *globalMinDist,
    int *globalMinIndex,
    int numBlocks
) {
    __shared__ int sDist[BLOCK_SIZE];
    __shared__ int sIndex[BLOCK_SIZE];

    int tid = threadIdx.x;
    int bestDist = INF;
    int bestIndex = -1;

    for (int i = tid; i < numBlocks; i += blockDim.x) {
        int d = blockMinDist[i];
        int idx = blockMinIndex[i];

        if (d < bestDist) {
            bestDist = d;
            bestIndex = idx;
        }
    }

    sDist[tid] = bestDist;
    sIndex[tid] = bestIndex;
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

__global__ void update_kernel(
    const int *graph,
    int *dist,
    unsigned char *visited,
    const int *globalMinDist,
    const int *globalMinIndex,
    int n
) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= n) return;

    int u = *globalMinIndex;
    int uDist = *globalMinDist;

    if (u < 0) return;

    if (v == u) {
        visited[v] = 1;
        return;
    }

    if (visited[v]) return;

    int w = graph[(size_t)u * n + v];

    if (w < INF && uDist < INF && uDist + w < dist[v]) {
        dist[v] = uDist + w;
    }
}

void enqueue_cuda_dijkstra(
    cudaStream_t stream,
    int *d_graph,
    int *d_dist,
    unsigned char *d_visited,
    int *d_blockMinDist,
    int *d_blockMinIndex,
    int *d_globalMinDist,
    int *d_globalMinIndex,
    int n,
    int source
) {
    int numBlocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    init_kernel<<<numBlocks, BLOCK_SIZE, 0, stream>>>(
        d_graph,
        d_dist,
        d_visited,
        n,
        source
    );

    for (int i = 0; i < n; i++) {
        find_block_min_kernel<<<numBlocks, BLOCK_SIZE, 0, stream>>>(
            d_dist,
            d_visited,
            d_blockMinDist,
            d_blockMinIndex,
            n
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
            d_globalMinIndex,
            n
        );
    }
}

int main(void) {
    printf("  CUDA Graphs Dijkstra\n");
    printf("  Single-Source Shortest Paths\n\n");

    int gpu_count = 0;
    cudaError_t gpu_err = cudaGetDeviceCount(&gpu_count);
    if (gpu_err != cudaSuccess || gpu_count == 0) {
        fprintf(stderr, "No CUDA-capable GPU detected. Aborting.\n");
        return 1;
    }

    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    printf("CUDA Device: %s (Compute Capability %d.%d)\n", prop.name, prop.major, prop.minor);
    printf("Block Size: %d threads\n\n", BLOCK_SIZE);

    int n;
    printf("Enter number of nodes: ");
    if (scanf("%d", &n) != 1 || n < 2) {
        printf("Invalid number of nodes (must be >= 2).\n");
        return 1;
    }

    int source;
    printf("Enter source vertex: ");
    if (scanf("%d", &source) != 1 || source < 0 || source >= n) {
        printf("Invalid source vertex (must be 0 to %d).\n", n - 1);
        return 1;
    }

    printf("\nBuilding random graph\n");
    int density = (n <= 100) ? 40 : 20;
    printf("Nodes: %d   Source: %d   Density: %d%%   Seed: 42\n", n, source, density);

    Graph *g = graph_create(n);
    fill_random_edges(g, density, 42);
    int *graph = flatten_matrix(g->adj_matrix, n);

    int *serialDist = (int *)malloc((size_t)n * sizeof(int));
    int *cudaDist = (int *)malloc((size_t)n * sizeof(int));

    if (!serialDist || !cudaDist) {
        printf("Host malloc failed\n");
        return 1;
    }

    int numBlocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    int *d_graph;
    int *d_dist;
    unsigned char *d_visited;
    int *d_blockMinDist;
    int *d_blockMinIndex;
    int *d_globalMinDist;
    int *d_globalMinIndex;

    CHECK_CUDA(cudaMalloc(&d_graph, (size_t)n * n * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_dist, (size_t)n * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_visited, (size_t)n * sizeof(unsigned char)));
    CHECK_CUDA(cudaMalloc(&d_blockMinDist, (size_t)numBlocks * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_blockMinIndex, (size_t)numBlocks * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_globalMinDist, sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_globalMinIndex, sizeof(int)));

    CHECK_CUDA(cudaMemcpy(
        d_graph,
        graph,
        (size_t)n * n * sizeof(int),
        cudaMemcpyHostToDevice
    ));

    double serialTotal = 0.0;

    for (int r = 0; r < RUNS; r++) {
        double t1 = now_seconds();
        serial_dijkstra(graph, serialDist, n, source);
        double t2 = now_seconds();
        serialTotal += t2 - t1;
    }

    cudaStream_t stream;
    cudaGraph_t graphObj;
    cudaGraphExec_t graphExec;

    CHECK_CUDA(cudaStreamCreate(&stream));
    CHECK_CUDA(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));

    enqueue_cuda_dijkstra(
        stream,
        d_graph,
        d_dist,
        d_visited,
        d_blockMinDist,
        d_blockMinIndex,
        d_globalMinDist,
        d_globalMinIndex,
        n,
        source
    );

    CHECK_CUDA(cudaStreamEndCapture(stream, &graphObj));
    CHECK_CUDA(cudaGraphInstantiate(&graphExec, graphObj, NULL, NULL, 0));

    CHECK_CUDA(cudaGraphLaunch(graphExec, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    float cudaTotalMs = 0.0f;

    for (int r = 0; r < RUNS; r++) {
        CHECK_CUDA(cudaEventRecord(start, stream));
        CHECK_CUDA(cudaGraphLaunch(graphExec, stream));
        CHECK_CUDA(cudaEventRecord(stop, stream));
        CHECK_CUDA(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
        cudaTotalMs += ms;
    }

    CHECK_CUDA(cudaMemcpy(
        cudaDist,
        d_dist,
        (size_t)n * sizeof(int),
        cudaMemcpyDeviceToHost
    ));

    int correct = 1;
    for (int i = 0; i < n; i++) {
        if (serialDist[i] != cudaDist[i]) {
            correct = 0;
            printf("Mismatch at vertex %d: CPU = %d, CUDA = %d\n",
                   i,
                   serialDist[i],
                   cudaDist[i]);
            break;
        }
    }

    double serialAvg = serialTotal / RUNS;
    double cudaAvg = (cudaTotalMs / RUNS) / 1000.0;

    printf("\n=== CUDA Graphs Dijkstra Results ===\n");
    printf("N: %d\n", n);
    printf("Source: %d\n", source);
    printf("Runs: %d\n", RUNS);
    printf("Serial avg: %.6f sec\n", serialAvg);
    printf("CUDA Graph avg: %.6f sec\n", cudaAvg);
    printf("Speedup: %.3fx\n", serialAvg / (cudaAvg > 0.0 ? cudaAvg : 1e-12));
    printf("Distances match: %s\n", correct ? "YES" : "NO");

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaGraphExecDestroy(graphExec));
    CHECK_CUDA(cudaGraphDestroy(graphObj));
    CHECK_CUDA(cudaStreamDestroy(stream));

    CHECK_CUDA(cudaFree(d_graph));
    CHECK_CUDA(cudaFree(d_dist));
    CHECK_CUDA(cudaFree(d_visited));
    CHECK_CUDA(cudaFree(d_blockMinDist));
    CHECK_CUDA(cudaFree(d_blockMinIndex));
    CHECK_CUDA(cudaFree(d_globalMinDist));
    CHECK_CUDA(cudaFree(d_globalMinIndex));

    free(graph);
    free(serialDist);
    free(cudaDist);
    graph_free(g);

    return 0;
}
