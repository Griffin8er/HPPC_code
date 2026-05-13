#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <limits.h>
#include <time.h>
#include <omp.h>

#define INF 1000000000

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

/* ================= 1. FW SEQUENTIAL (BASELINE) ================= */

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

/* ================= 2. FW RECURSIVE (CACHE-OBLIVIOUS) ================= */
/* Athens paper Algorithm 2: FW_SR with quadrant division */

static void fwi(int **dist, int row_start, int row_end, int col_start, int col_end,
                int k_start, int k_end) {
    // Base iteration: standard FW on the given submatrix and k-range
    for (int k = k_start; k < k_end; k++) {
        for (int i = row_start; i < row_end; i++) {
            for (int j = col_start; j < col_end; j++) {
                if (dist[i][k] < INF && dist[k][j] < INF) {
                    int new_dist = dist[i][k] + dist[k][j];
                    if (new_dist < dist[i][j]) {
                        dist[i][j] = new_dist;
                    }
                }
            }
        }
    }
}

static void fwr(int **dist, int row_start, int row_end, int col_start, int col_end,
                int k_start, int k_end, int base_size) {
    int row_size = row_end - row_start;
    int col_size = col_end - col_start;
    int k_size = k_end - k_start;

    // Base case: if any dimension is small enough, use iterative FW
    if (row_size <= base_size || col_size <= base_size || k_size <= base_size) {
        fwi(dist, row_start, row_end, col_start, col_end, k_start, k_end);
        return;
    }

    // Divide into quadrants
    int row_mid = row_start + row_size / 2;
    int col_mid = col_start + col_size / 2;
    int k_mid = k_start + k_size / 2;

    // Recursive calls in the order specified by Athens Algorithm 2
    // First four calls: top-left to bottom-right
    fwr(dist, row_start, row_mid, col_start, col_mid, k_start, k_mid, base_size);    // A11
    fwr(dist, row_start, row_mid, col_mid, col_end, k_start, k_mid, base_size);      // A12
    fwr(dist, row_mid, row_end, col_start, col_mid, k_start, k_mid, base_size);      // A21
    fwr(dist, row_mid, row_end, col_mid, col_end, k_start, k_mid, base_size);        // A22

    // Last four calls: reverse order (bottom-right to top-left)
    fwr(dist, row_mid, row_end, col_mid, col_end, k_mid, k_end, base_size);          // A22
    fwr(dist, row_mid, row_end, col_start, col_mid, k_mid, k_end, base_size);        // A21
    fwr(dist, row_start, row_mid, col_mid, col_end, k_mid, k_end, base_size);        // A12
    fwr(dist, row_start, row_mid, col_start, col_mid, k_mid, k_end, base_size);      // A11
}

int** fw_recursive(Graph *g, int base_size) {
    int n = g->vertices;
    int **dist = matrix_copy(g->adj_matrix, n);
    fwr(dist, 0, n, 0, n, 0, n, base_size);
    return dist;
}

/* ================= 3. FW TILED (CACHE-OPTIMIZED) ================= */
/* Sequential 3-phase blocked algorithm for cache locality */

int** fw_tiled(Graph *g, int B) {
    int n = g->vertices;
    int **dist = matrix_copy(g->adj_matrix, n);

    // Iterate over block rows/columns
    for (int kk = 0; kk < n; kk += B) {
        int k_end = (kk + B < n) ? kk + B : n;

        /* PHASE 1: Update diagonal block (kk, kk) */
        for (int k = kk; k < k_end; k++) {
            for (int i = kk; i < k_end; i++) {
                int dik = dist[i][k];
                if (dik >= INF) continue;
                for (int j = kk; j < k_end; j++) {
                    int dkj = dist[k][j];
                    if (dkj < INF) {
                        int new_dist = dik + dkj;
                        if (new_dist < dist[i][j]) {
                            dist[i][j] = new_dist;
                        }
                    }
                }
            }
        }

        /* PHASE 2: Update row and column blocks adjacent to diagonal */
        for (int bb = 0; bb < n; bb += B) {
            if (bb == kk) continue; // Skip diagonal block

            int b_end = (bb + B < n) ? bb + B : n;

            // Update row block (bb, kk)
            for (int k = kk; k < k_end; k++) {
                for (int i = bb; i < b_end; i++) {
                    int dik = dist[i][k];
                    if (dik >= INF) continue;
                    for (int j = kk; j < k_end; j++) {
                        int dkj = dist[k][j];
                        if (dkj < INF) {
                            int new_dist = dik + dkj;
                            if (new_dist < dist[i][j]) {
                                dist[i][j] = new_dist;
                            }
                        }
                    }
                }
            }

            // Update column block (kk, bb)
            for (int k = kk; k < k_end; k++) {
                for (int i = kk; i < k_end; i++) {
                    int dik = dist[i][k];
                    if (dik >= INF) continue;
                    for (int j = bb; j < b_end; j++) {
                        int dkj = dist[k][j];
                        if (dkj < INF) {
                            int new_dist = dik + dkj;
                            if (new_dist < dist[i][j]) {
                                dist[i][j] = new_dist;
                            }
                        }
                    }
                }
            }
        }

        /* PHASE 3: Update all remaining blocks */
        for (int ii = 0; ii < n; ii += B) {
            if (ii == kk) continue;
            int i_end = (ii + B < n) ? ii + B : n;

            for (int jj = 0; jj < n; jj += B) {
                if (jj == kk) continue;
                int j_end = (jj + B < n) ? jj + B : n;

                for (int k = kk; k < k_end; k++) {
                    for (int i = ii; i < i_end; i++) {
                        int dik = dist[i][k];
                        if (dik >= INF) continue;
                        for (int j = jj; j < j_end; j++) {
                            int dkj = dist[k][j];
                            if (dkj < INF) {
                                int new_dist = dik + dkj;
                                if (new_dist < dist[i][j]) {
                                    dist[i][j] = new_dist;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    return dist;
}

/* ================= 4. OMP FW NAIVE PARALLEL ================= */
/* Simple parallelization of i,j loops - naive approach */

int** omp_fw_naive(Graph *g) {
    int n = g->vertices;
    int **dist = matrix_copy(g->adj_matrix, n);

    for (int k = 0; k < n; k++) {
        #pragma omp parallel for collapse(2) schedule(static) shared(dist, k)
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

/* ================= 5. OMP FW TILED (3-PHASE BLOCKED) ================= */
/* 3-phase blocked algorithm with OpenMP parallelization */

int** omp_fw_tiled(Graph *g, int B) {
    int n = g->vertices;
    int **dist = matrix_copy(g->adj_matrix, n);

    for (int kk = 0; kk < n; kk += B) {
        int k_end = (kk + B < n) ? kk + B : n;

        /* PHASE 1: Diagonal block (sequential - dependency critical) */
        for (int k = kk; k < k_end; k++) {
            for (int i = kk; i < k_end; i++) {
                int dik = dist[i][k];
                if (dik >= INF) continue;
                for (int j = kk; j < k_end; j++) {
                    int dkj = dist[k][j];
                    if (dkj < INF) {
                        int new_dist = dik + dkj;
                        if (new_dist < dist[i][j]) {
                            dist[i][j] = new_dist;
                        }
                    }
                }
            }
        }


        /* PHASE 2: Row and column blocks (each block gets its own parallel task) */
        // Calculate number of cross blocks (excluding diagonal)
        int num_blocks = (n + B - 1) / B;
        int num_cross_blocks = num_blocks - 1;  // Exclude diagonal block

        // Process all cross blocks in parallel: each row/column block is independent
        #pragma omp parallel for schedule(static)
        for (int idx = 0; idx < 2 * num_cross_blocks; idx++) {
            int block_idx = idx % num_cross_blocks;
            int bb = (block_idx >= kk / B) ? (block_idx + 1) * B : block_idx * B;
            int b_end = (bb + B < n) ? bb + B : n;

            if (idx < num_cross_blocks) {
                // Process ROW block (bb, kk)
                for (int k = kk; k < k_end; k++) {
                    for (int i = bb; i < b_end; i++) {
                        int dik = dist[i][k];
                        if (dik >= INF) continue;
                        for (int j = kk; j < k_end; j++) {
                            int dkj = dist[k][j];
                            if (dkj < INF) {
                                int new_dist = dik + dkj;
                                if (new_dist < dist[i][j]) {
                                    dist[i][j] = new_dist;
                                }
                            }
                        }
                    }
                }
            } else {
                // Process COLUMN block (kk, bb)
                for (int k = kk; k < k_end; k++) {
                    for (int i = kk; i < k_end; i++) {
                        int dik = dist[i][k];
                        if (dik >= INF) continue;
                        for (int j = bb; j < b_end; j++) {
                            int dkj = dist[k][j];
                            if (dkj < INF) {
                                int new_dist = dik + dkj;
                                if (new_dist < dist[i][j]) {
                                    dist[i][j] = new_dist;
                                }
                            }
                        }
                    }
                }
            }
        }

        /* PHASE 3: Remaining blocks (parallel over block grid) */
        #pragma omp parallel for collapse(2) schedule(static)
        for (int ii = 0; ii < n; ii += B) {
            for (int jj = 0; jj < n; jj += B) {
                if (ii == kk || jj == kk) continue;

                int i_end = (ii + B < n) ? ii + B : n;
                int j_end = (jj + B < n) ? jj + B : n;

                for (int k = kk; k < k_end; k++) {
                    for (int i = ii; i < i_end; i++) {
                        int dik = dist[i][k];
                        if (dik >= INF) continue;
                        for (int j = jj; j < j_end; j++) {
                            int dkj = dist[k][j];
                            if (dkj < INF) {
                                int new_dist = dik + dkj;
                                if (new_dist < dist[i][j]) {
                                    dist[i][j] = new_dist;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    return dist;
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

/* ================= BLOCK-SIZE VALIDATOR ================= */

int is_valid_block_size(int B) {
    static const int allowed[] = {8, 16, 32, 64, 128, 256, 512};
    int count = sizeof(allowed) / sizeof(allowed[0]);
    for (int i = 0; i < count; i++) {
        if (allowed[i] == B) return 1;
    }
    return 0;
}

/* ================= MAIN ================= */

int main() {
    printf("=======================================================\n");
    printf("  FLOYD-WARSHALL ALGORITHM COMPARISON\n");
    printf("  All-Pairs Shortest Paths (APSP)\n");
    printf("  Athens Paper Implementation\n");
    printf("=======================================================\n\n");

    /* Read user input */
    int n, B;

    printf("Enter number of nodes: ");
    if (scanf("%d", &n) != 1 || n < 2) {
        printf("Invalid number of nodes (must be >= 2).\n");
        return 1;
    }

    printf("Enter block size [8, 16, 32, 64, 128, 256, 512]: ");
    if (scanf("%d", &B) != 1 || !is_valid_block_size(B)) {
        printf("Invalid block size. Must be one of: 8, 16, 32, 64, 128, 256, 512.\n");
        return 1;
    }

    printf("\n--- Building random graph ---\n");
    printf("Nodes: %d   Block size: %d\n", n, B);

    /* Build random graph */
    Graph *g = graph_create(n);
    int density = (n <= 100) ? 40 : 20;
    fill_random_edges(g, density, 42);

    /* Time each algorithm */
    double t0, t1;
    double time_seq, time_rec, time_tiled, time_omp_naive, time_omp_tiled;

    printf("\n--- Running algorithms ---\n");

    /* 1. FW Sequential */
    printf("Running FW Sequential...\n");
    t0 = omp_get_wtime();
    int **result_seq = fw_sequential(g);
    t1 = omp_get_wtime();
    time_seq = (t1 - t0) * 1000.0;

    /* 2. FW Recursive */
    printf("Running FW Recursive...\n");
    t0 = omp_get_wtime();
    int **result_rec = fw_recursive(g, B);
    t1 = omp_get_wtime();
    time_rec = (t1 - t0) * 1000.0;

    /* 3. FW Tiled */
    printf("Running FW Tiled...\n");
    t0 = omp_get_wtime();
    int **result_tiled = fw_tiled(g, B);
    t1 = omp_get_wtime();
    time_tiled = (t1 - t0) * 1000.0;

    /* 4. OMP FW Naive */
    printf("Running OMP FW Naive...\n");
    t0 = omp_get_wtime();
    int **result_omp_naive = omp_fw_naive(g);
    t1 = omp_get_wtime();
    time_omp_naive = (t1 - t0) * 1000.0;

    /* 5. OMP FW Tiled */
    printf("Running OMP FW Tiled (3-Phase)...\n");
    t0 = omp_get_wtime();
    int **result_omp_tiled = omp_fw_tiled(g, B);
    t1 = omp_get_wtime();
    time_omp_tiled = (t1 - t0) * 1000.0;

    /* Correctness check */
    int ok_rec = matrices_equal(result_seq, result_rec, n);
    int ok_tiled = matrices_equal(result_seq, result_tiled, n);
    int ok_omp_naive = matrices_equal(result_seq, result_omp_naive, n);
    int ok_omp_tiled = matrices_equal(result_seq, result_omp_tiled, n);

    printf("\n--- Correctness Check (APSP for all %d×%d pairs) ---\n", n, n);
    printf("  FW Recursive           : %s\n", ok_rec ? "PASS" : "FAIL");
    printf("  FW Tiled               : %s\n", ok_tiled ? "PASS" : "FAIL");
    printf("  OMP FW Naive           : %s\n", ok_omp_naive ? "PASS" : "FAIL");
    printf("  OMP FW Tiled           : %s\n", ok_omp_tiled ? "PASS" : "FAIL");

    /* Runtime comparison */
    double base = time_seq > 0 ? time_seq : 1e-9;

    printf("\n=======================================================\n");
    printf("       RUNTIME COMPARISON (block size = %d)\n", B);
    printf("=======================================================\n");
    printf(" %-28s | %-12s | %-10s\n", "Algorithm", "Time (ms)", "Speedup");
    printf("-------------------------------------------------------\n");
    printf(" %-28s | %12.4f | %9.2fx\n", "FW Sequential", time_seq, 1.0);
    printf(" %-28s | %12.4f | %9.2fx\n", "FW Recursive", time_rec, base / (time_rec > 0 ? time_rec : 1e-9));
    printf(" %-28s | %12.4f | %9.2fx\n", "FW Tiled", time_tiled, base / (time_tiled > 0 ? time_tiled : 1e-9));
    printf(" %-28s | %12.4f | %9.2fx\n", "OMP FW Naive", time_omp_naive, base / (time_omp_naive > 0 ? time_omp_naive : 1e-9));
    printf(" %-28s | %12.4f | %9.2fx\n", "OMP FW Tiled", time_omp_tiled, base / (time_omp_tiled > 0 ? time_omp_tiled : 1e-9));
    printf("=======================================================\n");
    printf(" Note: Speedup relative to FW Sequential (baseline).\n");
    printf(" All algorithms compute APSP for all %d×%d node pairs.\n", n, n);
    printf("=======================================================\n");

    /* Cleanup */
    matrix_free(result_seq, n);
    matrix_free(result_rec, n);
    matrix_free(result_tiled, n);
    matrix_free(result_omp_naive, n);
    matrix_free(result_omp_tiled, n);
    graph_free(g);

    return 0;
}
