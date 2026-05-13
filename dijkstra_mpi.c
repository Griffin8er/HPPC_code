#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <mpi.h>

#define N 8192
#define SOURCE 1
#define MAXINT 9999999
#define RUNS 25

#define EDGE_PROBABILITY 25
#define MAX_WEIGHT 20

void build_random_graph(int graph[N][N]) {

    int i, j;

    srand(42);

    for (i = 0; i < N; i++) {

        for (j = 0; j < N; j++) {

            if (i == j) {
                graph[i][j] = 0;
            }
            else {

                int r = rand() % 100;

                if (r < EDGE_PROBABILITY) {
                    graph[i][j] =
                        1 + rand() % MAX_WEIGHT;
                }
                else {
                    graph[i][j] = MAXINT;
                }
            }
        }
    }
}


double now_seconds(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec +
           (double)tv.tv_usec / 1000000.0;
}

void serial_dijkstra(int graph[N][N],
                     int source,
                     int distance[N]) {

    int visited[N];
    int i, count;
    int nextNode;
    int minDistance;

    for (i = 0; i < N; i++) {
        distance[i] = graph[source][i];
        visited[i] = 0;
    }

    distance[source] = 0;
    visited[source] = 1;
    count = 1;

    while (count < N) {

        minDistance = MAXINT;

        for (i = 0; i < N; i++) {
            if (!visited[i] &&
                distance[i] < minDistance) {

                minDistance = distance[i];
                nextNode = i;
            }
        }

        visited[nextNode] = 1;
        count++;

        for (i = 0; i < N; i++) {
            if (!visited[i] &&
                graph[nextNode][i] < MAXINT &&
                minDistance +
                graph[nextNode][i] <
                distance[i]) {

                distance[i] =
                    minDistance +
                    graph[nextNode][i];
            }
        }
    }
}

void mpi_dijkstra(int n,
                  int source,
                  int *localWeight,
                  int *localDistance,
                  MPI_Comm comm) {

    int i, j;
    int nlocal;
    int firstvtx;
    int u, udist;

    int lminpair[2];
    int gminpair[2];

    int npes, myrank;
    int *marker;

    MPI_Comm_size(comm, &npes);
    MPI_Comm_rank(comm, &myrank);

    nlocal = n / npes;
    firstvtx = myrank * nlocal;

    marker = malloc(nlocal * sizeof(int));

    for (j = 0; j < nlocal; j++) {
        localDistance[j] =
            localWeight[source * nlocal + j];

        marker[j] = 1;
    }

    if (source >= firstvtx &&
        source < firstvtx + nlocal) {

        marker[source - firstvtx] = 0;
        localDistance[source - firstvtx] = 0;
    }

    for (i = 1; i < n; i++) {

        lminpair[0] = MAXINT;
        lminpair[1] = -1;

        for (j = 0; j < nlocal; j++) {
            if (marker[j] &&
                localDistance[j] <
                lminpair[0]) {

                lminpair[0] =
                    localDistance[j];

                lminpair[1] =
                    firstvtx + j;
            }
        }

        MPI_Allreduce(
            lminpair,
            gminpair,
            1,
            MPI_2INT,
            MPI_MINLOC,
            comm
        );

        udist = gminpair[0];
        u = gminpair[1];

        if (u >= firstvtx &&
            u < firstvtx + nlocal) {

            marker[u - firstvtx] = 0;
        }

        for (j = 0; j < nlocal; j++) {

            if (marker[j] &&
                localWeight[u * nlocal + j]
                < MAXINT &&
                udist +
                localWeight[u * nlocal + j]
                < localDistance[j]) {

                localDistance[j] =
                    udist +
                    localWeight[u * nlocal + j];
            }
        }
    }

    free(marker);
}

int main(int argc, char *argv[]) {

    int npes, myrank;
    int nlocal;

    int (*weight)[N] = NULL;

    int *sendbuf = NULL;
    int *localWeight;
    int *localDistance;
    int *mpiDistance = NULL;

    int serialDistance[N];

    int i, j, k;

    double serialTotal = 0.0;
    double mpiTotal = 0.0;

    MPI_Init(&argc, &argv);

    MPI_Comm_size(
        MPI_COMM_WORLD,
        &npes);

    MPI_Comm_rank(
        MPI_COMM_WORLD,
        &myrank);

    if (N % npes != 0) {
        if (myrank == 0)
            printf("N must divide processes\n");

        MPI_Finalize();
        return 1;
    }

    nlocal = N / npes;

    localWeight =
        malloc(N * nlocal * sizeof(int));

    localDistance =
        malloc(nlocal * sizeof(int));

    if (myrank == 0) {

        weight =
            malloc(N * sizeof(*weight));

        sendbuf =
            malloc(N * N * sizeof(int));

        mpiDistance =
            malloc(N * sizeof(int));

        
        build_random_graph(weight);


        /* SERIAL RUNS */

        for (k = 0; k < RUNS; k++) {

            double t1 =
                now_seconds();

            serial_dijkstra(
                weight,
                SOURCE,
                serialDistance
            );

            double t2 =
                now_seconds();

            serialTotal +=
                (t2 - t1);
        }

        /* Prepare scatter */

        for (int p = 0; p < npes; p++) {
            for (i = 0; i < N; i++) {
                for (j = 0; j < nlocal; j++) {

                    sendbuf[
                        p * N * nlocal +
                        i * nlocal +
                        j
                    ] =
                        weight[i][p * nlocal + j];
                }
            }
        }
    }

    /* MPI RUNS */

    for (k = 0; k < RUNS; k++) {

        MPI_Barrier(
            MPI_COMM_WORLD);

        double t1 =
            MPI_Wtime();

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

        mpi_dijkstra(
            N,
            SOURCE,
            localWeight,
            localDistance,
            MPI_COMM_WORLD
        );

        MPI_Gather(
            localDistance,
            nlocal,
            MPI_INT,
            mpiDistance,
            nlocal,
            MPI_INT,
            0,
            MPI_COMM_WORLD
        );

        MPI_Barrier(
            MPI_COMM_WORLD);

        double t2 =
            MPI_Wtime();

        if (myrank == 0)
            mpiTotal += (t2 - t1);
    }

    if (myrank == 0) {

        double serialAvg =
            serialTotal / RUNS;

        double mpiAvg =
            mpiTotal / RUNS;

        printf("\n=== Average Results ===\n");
        printf("Runs: %d\n", RUNS);
        printf("Processes: %d\n", npes);

        printf("Serial avg: %.6f sec\n",
               serialAvg);

        printf("MPI avg: %.6f sec\n",
               mpiAvg);

        printf("Speedup: %.3fx\n",
               serialAvg / mpiAvg);
    }

    MPI_Finalize();

    return 0;
}