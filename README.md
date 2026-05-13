# Optimization of All-Pairs Shortest Path Algorithms

## Parallel Dijkstra Code
Built off this paper: [Parallel Dijkstra](https://repository.stcloudstate.edu/cgi/viewcontent.cgi?article=1044&context=csit_etds)

`dijkstra_mpi.c`:
- Direct implementation of code from paper

`dijkstra_cuda.cu`:
- SSSP implementation of cuda implementation of paper

`parallel_dijkstra.cu`:
- APSP implementation of sequential Dijkstra, parallel Dijkstra, and cuda Dijkstra implementations

## Delta Stepping
Referenced Papers:
- K. Wang, D. S. Fussell, and C. Lin, “A fast work-efficient SSSP algorithm for GPUs,” Feb. 2021, doi: https://doi.org/10.1145/3437801.3441605.
- U. Sridhar, M. P. Blanco, R. Mayuranath, D. G. Spampinato, T. M. Low, and S. McMillan, “Delta-Stepping SSSP: From Vertices and Edges to GraphBLAS Implementations,” 2019 IEEE International Parallel and Distributed Processing Symposium Workshops (IPDPSW), pp. 241–250, May 2019, doi: https://doi.org/10.1109/ipdpsw.2019.00047. 

Referenced Code:
- [GraphBlas SSSP Algorithms](https://github.com/cmu-sei/gbtl/tree/master) 
- [CUDA Delta Stepping](https://github.com/naizhengtan/Delta-stepping--in-CUDA)

## Floyd-Warshall
Referenced Athens paper: [Floyd-Warshall](https://github.com/user-attachments/files/27681116/floyd-warshall.pdf)

`Floyd_Warshall_CPU.c`:
- OpenMP Implementation based off pseudocode from paper

`Floyd_Warshall_GPU.cu`:
- Based off OpenMP implementation from paper, translated to GPU

