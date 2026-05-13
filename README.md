# Optimization of All-Pairs Shortest Path Algorithms

## Parallel Dijkstra Code
Built off this paper: [Parallel Dijkstra](https://repository.stcloudstate.edu/cgi/viewcontent.cgi?article=1044&context=csit_etds)

`Dijkstra_mpi.c`:
- Direct implementation of code from paper

`Dijkstra_cuda.cu`:
- SSSP implementation of cuda implementation of paper

`Parallel_Dijkstra.cu`:
- APSP implementation of sequential Dijkstra, parallel Dijkstra, and cuda Dijkstra implementations\

## Delta Stepping

## Floyd-Warshall
Referenced Athens paper: [Floyd-Warshall](https://github.com/user-attachments/files/27681116/floyd-warshall.pdf)

`Floyd_Warshall_CPU.c`:
- OpenMP Implementation based off pseudocode from paper

`Floyd_Warshall_GPU.cu`:
- Based off OpenMP implementation from paper, translated to GPU

