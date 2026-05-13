#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>

#define INF 1000000000

/* CUDA error-checking macro */
#define CUDA_CHECK(call) do {                                            \
    cudaError_t err__ = (call);                                          \
    if (err__ != cudaSuccess) {                                          \
        fprintf(stderr, "CUDA error %s:%d: %s\n",                        \
                __FILE__, __LINE__, cudaGetErrorString(err__));          \
        exit(1);                                                         \
    }                                                                    \
} while (0)

typedef struct {
    int vertices;
    int **adj_matrix;
} Graph;

/* ================= GRAPH UTILITIES ================= */

Graph* graph_create(int vertices) {
    Graph *g = (Graph *)malloc(sizeof(Graph));
    g->vertices = vertices;
    g->adj_matrix = (int **)malloc(vertices * sizeof(int *));
    for (int i = 0; i < vertices; i++) {
        g->adj_matrix[i] = (int *)malloc(vertices * sizeof(int));
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
    for (int i = 0; i < g->vertices; i++) free(g->adj_matrix[i]);
    free(g->adj_matrix);
    free(g);
}

int** matrix_copy(int **src, int n) {
    int **dst = (int **)malloc(n * sizeof(int *));
    for (int i = 0; i < n; i++) {
        dst[i] = (int *)malloc(n * sizeof(int));
        memcpy(dst[i], src[i], n * sizeof(int));
    }
    return dst;
}

void matrix_free(int **m, int n) {
    for (int i = 0; i < n; i++) free(m[i]);
    free(m);
}

int matrices_equal(int **a, int **b, int n) {
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            if (a[i][j] != b[i][j]) return 0;
    return 1;
}

/* Flatten row-pointer matrix into contiguous array for GPU transfer */
int* flatten_matrix(int **m, int n) {
    int *flat = (int *)malloc((size_t)n * n * sizeof(int));
    for (int i = 0; i < n; i++) {
        memcpy(flat + (size_t)i * n, m[i], n * sizeof(int));
    }
    return flat;
}

/* Convert flat array back to row-pointer format */
int** unflatten_matrix(int *flat, int n) {
    int **m = (int **)malloc(n * sizeof(int *));
    for (int i = 0; i < n; i++) {
        m[i] = (int *)malloc(n * sizeof(int));
        memcpy(m[i], flat + (size_t)i * n, n * sizeof(int));
    }
    return m;
}

/* ================= FW SEQUENTIAL (for correctness check) ================= */

int** fw_sequential(Graph *g) {
    int n = g->vertices;
    int **dist = matrix_copy(g->adj_matrix, n);

    for (int k = 0; k < n; k++) {
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (dist[i][k] < INF && dist[k][j] < INF) {
                    int new_dist = dist[i][k] + dist[k][j];
                    if (new_dist < dist[i][j]) {
                        dist[i][j] = new_dist;
                    }
                }
            }
        }
    }

    return dist;
}

/* ================= CUDA IMPLEMENTATIONS ================= */

/* CUDA tile dimension: 16x16 = 256 threads per block */
#define CUDA_TILE 16

/* ================= 1. CUDA FW NAIVE ================= */
/* Simple parallelization: one thread per (i,j) element, no shared memory */

__global__ void cuda_fw_naive_kernel(int *dist, int n, int k) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= n || j >= n) return;

    int dik = dist[i * n + k];
    int dkj = dist[k * n + j];

    if (dik < INF && dkj < INF) {
        int new_dist = dik + dkj;
        int dij = dist[i * n + j];
        if (new_dist < dij) {
            dist[i * n + j] = new_dist;
        }
    }
}

int** cuda_fw_naive(Graph *g) {
    int n = g->vertices;

    /* Flatten host matrix */
    int *h_dist = flatten_matrix(g->adj_matrix, n);

    /* Allocate and copy to device */
    int *d_dist;
    size_t bytes = (size_t)n * n * sizeof(int);
    CUDA_CHECK(cudaMalloc(&d_dist, bytes));
    CUDA_CHECK(cudaMemcpy(d_dist, h_dist, bytes, cudaMemcpyHostToDevice));

    dim3 block(CUDA_TILE, CUDA_TILE);
    dim3 grid((n + CUDA_TILE - 1) / CUDA_TILE,
              (n + CUDA_TILE - 1) / CUDA_TILE);

    /* One kernel launch per k iteration */
    for (int k = 0; k < n; k++) {
        cuda_fw_naive_kernel<<<grid, block>>>(d_dist, n, k);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    /* Copy back to host */
    CUDA_CHECK(cudaMemcpy(h_dist, d_dist, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_dist));

    int **result = unflatten_matrix(h_dist, n);
    free(h_dist);
    return result;
}

/* ================= 2. CUDA FW TILED (3-PHASE BLOCKED) ================= */
/* Matches the Athens paper's blocked algorithm with shared memory optimization */

/* PHASE 1: Diagonal block - must be sequential within block */
__global__ void cuda_fw_phase1(int *dist, int n, int block_k) {
    __shared__ int tile[CUDA_TILE][CUDA_TILE];

    int ty = threadIdx.y;
    int tx = threadIdx.x;
    int i = block_k * CUDA_TILE + ty;
    int j = block_k * CUDA_TILE + tx;

    /* Load tile into shared memory */
    if (i < n && j < n)
        tile[ty][tx] = dist[i * n + j];
    else
        tile[ty][tx] = INF;
    __syncthreads();

    /* Process diagonal block: sequential k-loop within the tile */
    #pragma unroll
    for (int k = 0; k < CUDA_TILE; k++) {
        int dik = tile[ty][k];
        int dkj = tile[k][tx];
        if (dik < INF && dkj < INF) {
            int new_dist = dik + dkj;
            if (new_dist < tile[ty][tx]) {
                tile[ty][tx] = new_dist;
            }
        }
        __syncthreads();
    }

    /* Write back to global memory */
    if (i < n && j < n)
        dist[i * n + j] = tile[ty][tx];
}

/* PHASE 2: Row and column blocks sharing the diagonal */
__global__ void cuda_fw_phase2(int *dist, int n, int block_k) {
    if (blockIdx.x == block_k) return; /* Skip diagonal block */

    __shared__ int tile_diag[CUDA_TILE][CUDA_TILE];
    __shared__ int tile_self[CUDA_TILE][CUDA_TILE];

    int ty = threadIdx.y;
    int tx = threadIdx.x;

    /* Load diagonal tile */
    int i_diag = block_k * CUDA_TILE + ty;
    int j_diag = block_k * CUDA_TILE + tx;
    if (i_diag < n && j_diag < n)
        tile_diag[ty][tx] = dist[i_diag * n + j_diag];
    else
        tile_diag[ty][tx] = INF;

    /* Load row or column tile based on blockIdx.y */
    int i_self, j_self;
    if (blockIdx.y == 0) {
        /* ROW BLOCK: (block_k, blockIdx.x) */
        i_self = block_k * CUDA_TILE + ty;
        j_self = blockIdx.x * CUDA_TILE + tx;
    } else {
        /* COLUMN BLOCK: (blockIdx.x, block_k) */
        i_self = blockIdx.x * CUDA_TILE + ty;
        j_self = block_k * CUDA_TILE + tx;
    }

    if (i_self < n && j_self < n)
        tile_self[ty][tx] = dist[i_self * n + j_self];
    else
        tile_self[ty][tx] = INF;

    __syncthreads();

    /* Compute updates using shared memory */
    #pragma unroll
    for (int k = 0; k < CUDA_TILE; k++) {
        int dik, dkj;
        if (blockIdx.y == 0) {
            /* Row block: i from diagonal, j from self */
            dik = tile_diag[ty][k];
            dkj = tile_self[k][tx];
        } else {
            /* Column block: i from self, j from diagonal */
            dik = tile_self[ty][k];
            dkj = tile_diag[k][tx];
        }

        if (dik < INF && dkj < INF) {
            int new_dist = dik + dkj;
            if (new_dist < tile_self[ty][tx]) {
                tile_self[ty][tx] = new_dist;
            }
        }
        __syncthreads();
    }

    /* Write back */
    if (i_self < n && j_self < n)
        dist[i_self * n + j_self] = tile_self[ty][tx];
}

/* PHASE 3: All remaining blocks (independent) */
__global__ void cuda_fw_phase3(int *dist, int n, int block_k) {
    if (blockIdx.x == block_k || blockIdx.y == block_k) return;

    __shared__ int tile_row[CUDA_TILE][CUDA_TILE];
    __shared__ int tile_col[CUDA_TILE][CUDA_TILE];

    int ty = threadIdx.y;
    int tx = threadIdx.x;

    int i = blockIdx.y * CUDA_TILE + ty;
    int j = blockIdx.x * CUDA_TILE + tx;

    /* Load row tile: (block_k, blockIdx.x) */
    int i_row = block_k * CUDA_TILE + ty;
    int j_row = blockIdx.x * CUDA_TILE + tx;
    if (i_row < n && j_row < n)
        tile_row[ty][tx] = dist[i_row * n + j_row];
    else
        tile_row[ty][tx] = INF;

    /* Load column tile: (blockIdx.y, block_k) */
    int i_col = blockIdx.y * CUDA_TILE + ty;
    int j_col = block_k * CUDA_TILE + tx;
    if (i_col < n && j_col < n)
        tile_col[ty][tx] = dist[i_col * n + j_col];
    else
        tile_col[ty][tx] = INF;

    __syncthreads();

    /* Compute update */
    int dij;
    if (i < n && j < n)
        dij = dist[i * n + j];
    else
        dij = INF;

    #pragma unroll
    for (int k = 0; k < CUDA_TILE; k++) {
        int dik = tile_col[ty][k];
        int dkj = tile_row[k][tx];
        if (dik < INF && dkj < INF) {
            int new_dist = dik + dkj;
            if (new_dist < dij) {
                dij = new_dist;
            }
        }
    }

    /* Write back */
    if (i < n && j < n)
        dist[i * n + j] = dij;
}

int** cuda_fw_tiled(Graph *g) {
    int n = g->vertices;

    /* Flatten and copy to device */
    int *h_dist = flatten_matrix(g->adj_matrix, n);

    int *d_dist;
    size_t bytes = (size_t)n * n * sizeof(int);
    CUDA_CHECK(cudaMalloc(&d_dist, bytes));
    CUDA_CHECK(cudaMemcpy(d_dist, h_dist, bytes, cudaMemcpyHostToDevice));

    int num_blocks = (n + CUDA_TILE - 1) / CUDA_TILE;
    dim3 block(CUDA_TILE, CUDA_TILE);

    /* 3-phase blocked algorithm */
    for (int block_k = 0; block_k < num_blocks; block_k++) {
        /* Phase 1: Diagonal block */
        dim3 grid_phase1(1, 1);
        cuda_fw_phase1<<<grid_phase1, block>>>(d_dist, n, block_k);

        /* Phase 2: Row and column blocks */
        dim3 grid_phase2(num_blocks, 2);
        cuda_fw_phase2<<<grid_phase2, block>>>(d_dist, n, block_k);

        /* Phase 3: Remaining blocks */
        dim3 grid_phase3(num_blocks, num_blocks);
        cuda_fw_phase3<<<grid_phase3, block>>>(d_dist, n, block_k);
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    /* Copy back to host */
    CUDA_CHECK(cudaMemcpy(h_dist, d_dist, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_dist));

    int **result = unflatten_matrix(h_dist, n);
    free(h_dist);
    return result;
}

/* ================= RANDOM GRAPH GENERATOR ================= */

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

/* ================= MAIN ================= */

int main() {
    printf("=======================================================\n");
    printf("  CUDA FLOYD-WARSHALL IMPLEMENTATIONS\n");
    printf("  All-Pairs Shortest Paths (APSP)\n");
    printf("=======================================================\n\n");

    /* Check for CUDA device */
    int gpu_count = 0;
    cudaError_t gpu_err = cudaGetDeviceCount(&gpu_count);
    if (gpu_err != cudaSuccess || gpu_count == 0) {
        fprintf(stderr, "No CUDA-capable GPU detected. Aborting.\n");
        return 1;
    }

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("CUDA Device: %s (Compute Capability %d.%d)\n",
           prop.name, prop.major, prop.minor);
    printf("GPU Tile Size: %dx%d threads\n\n", CUDA_TILE, CUDA_TILE);

    /* Read user input */
    int n;

    printf("Enter number of nodes: ");
    if (scanf("%d", &n) != 1 || n < 2) {
        printf("Invalid number of nodes (must be >= 2).\n");
        return 1;
    }

    printf("\n--- Building random graph ---\n");
    printf("Nodes: %d   GPU Tile Size: %dx%d\n", n, CUDA_TILE, CUDA_TILE);

    /* Build random graph */
    Graph *g = graph_create(n);
    int density = (n <= 100) ? 40 : 20;
    fill_random_edges(g, density, 42);

    /* Time each algorithm */
    double t0, t1;
    double time_seq, time_cuda_naive, time_cuda_tiled;

    printf("\n--- Running algorithms ---\n");

    /* 1. FW Sequential (baseline for correctness) */
    printf("Running FW Sequential (CPU baseline)...\n");
    t0 = (double)clock() / CLOCKS_PER_SEC;
    int **result_seq = fw_sequential(g);
    t1 = (double)clock() / CLOCKS_PER_SEC;
    time_seq = (t1 - t0) * 1000.0;

    /* 2. CUDA FW Naive */
    printf("Running CUDA FW Naive...\n");
    t0 = (double)clock() / CLOCKS_PER_SEC;
    int **result_cuda_naive = cuda_fw_naive(g);
    t1 = (double)clock() / CLOCKS_PER_SEC;
    time_cuda_naive = (t1 - t0) * 1000.0;

    /* 3. CUDA FW Tiled (3-Phase) */
    printf("Running CUDA FW Tiled (3-Phase)...\n");
    t0 = (double)clock() / CLOCKS_PER_SEC;
    int **result_cuda_tiled = cuda_fw_tiled(g);
    t1 = (double)clock() / CLOCKS_PER_SEC;
    time_cuda_tiled = (t1 - t0) * 1000.0;

    /* Correctness check */
    int ok_cuda_naive = matrices_equal(result_seq, result_cuda_naive, n);
    int ok_cuda_tiled = matrices_equal(result_seq, result_cuda_tiled, n);

    printf("\n--- Correctness Check (APSP for all %d×%d pairs) ---\n", n, n);
    printf("  CUDA FW Naive          : %s\n", ok_cuda_naive ? "PASS" : "FAIL");
    printf("  CUDA FW Tiled          : %s\n", ok_cuda_tiled ? "PASS" : "FAIL");

    /* Runtime comparison */
    double base = time_seq > 0 ? time_seq : 1e-9;

    printf("\n=======================================================\n");
    printf("       RUNTIME COMPARISON (GPU tile = %dx%d)\n", CUDA_TILE, CUDA_TILE);
    printf("=======================================================\n");
    printf(" %-28s | %-12s | %-10s\n", "Algorithm", "Time (ms)", "Speedup");
    printf("-------------------------------------------------------\n");
    printf(" %-28s | %12.4f | %9.2fx\n", "FW Sequential (CPU)", time_seq, 1.0);
    printf(" %-28s | %12.4f | %9.2fx\n", "CUDA FW Naive", time_cuda_naive, base / (time_cuda_naive > 0 ? time_cuda_naive : 1e-9));
    printf(" %-28s | %12.4f | %9.2fx\n", "CUDA FW Tiled (3-Phase)", time_cuda_tiled, base / (time_cuda_tiled > 0 ? time_cuda_tiled : 1e-9));
    printf("=======================================================\n");
    printf(" Note: Speedup relative to FW Sequential (CPU baseline).\n");
    printf(" CUDA timings include H2D + compute + D2H transfers.\n");
    printf(" All algorithms compute APSP for all %d×%d node pairs.\n", n, n);
    printf("=======================================================\n");

    /* Cleanup */
    matrix_free(result_seq, n);
    matrix_free(result_cuda_naive, n);
    matrix_free(result_cuda_tiled, n);
    graph_free(g);

    return 0;
}
