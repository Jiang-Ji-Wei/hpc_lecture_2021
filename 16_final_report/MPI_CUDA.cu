/****************************************
氏名：JIANG JIWEI
学籍番号：21M30519
説明：
MPI＋CUDAの並列化を実現したいですけど、うまく行けませんでした。
Flopsが出ませんでした。
*****************************************/

#include <mpi.h>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <chrono>
using namespace std;

__global__ void matmul(float *A, float *B, float *C, int N, int size) {
  int i = blockIdx.x;
  int j = threadIdx.x;
  float sum = 0;
  for (int k=0; k<N; k++) {
    sum += A[N*i+k] * B[N/size*k+j];
  }
  C[N*i+j] = sum;
}

int main(int argc, char **argv) {
//init MPI
  int size, rank;
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &size);//number of process
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);//process number 

  int N = 256;
  int full_size = N * N * sizeof(float);
  int sub_size = N * N /size * sizeof(float);
  float *A, *B, *C, *subA, *subB, *subC, *recv;
  cudaMallocManaged(&A, full_size);
  cudaMallocManaged(&B, full_size);
  cudaMallocManaged(&C, full_size);
  cudaMallocManaged(&subA, sub_size);
  cudaMallocManaged(&subB, sub_size);
  cudaMallocManaged(&subC, sub_size);
  cudaMallocManaged(&recv, sub_size);
  for (int i=0; i<N; i++) {
    for (int j=0; j<N; j++) {
      A[N*i+j] = drand48();
      B[N*i+j] = drand48();
      C[N*i+j] = 0;
    }
  }
  //initial subMatrix in different rank
  int offset = N/size*rank; //dim N separate into size
  for (int i=0; i<N/size; i++)
    for (int j=0; j<N; j++)
      subA[N*i+j] = A[N*(i+offset)+j];//Seperate A by row
  for (int i=0; i<N; i++)
    for (int j=0; j<N/size; j++)
      subB[N/size*i+j] = B[N*i+j+offset];//Seperate B by column
  for (int i = 0; i < sub_size; i++){
      subC[i] = 0;
  }
  int recv_from = (rank + 1) % size;
  int send_to = (rank - 1 + size) % size;

  double comp_time = 0, comm_time = 0;
  for(int irank=0; irank<size; irank++) {
    auto tic = chrono::steady_clock::now();//time

    offset = N/size*((rank+irank) % size);
    matmul<<<N/size,N/size>>>(subA, subB, &subC[offset], N, size);
    cudaDeviceSynchronize();

    auto toc = chrono::steady_clock::now();//time
    comp_time += chrono::duration<double>(toc - tic).count();//computation time?
    MPI_Request request[2];
    MPI_Isend(&subB[0], N*N/size, MPI_FLOAT, send_to, 0, MPI_COMM_WORLD, &request[0]);
    MPI_Irecv(&recv[0], N*N/size, MPI_FLOAT, recv_from, 0, MPI_COMM_WORLD, &request[1]);
    MPI_Waitall(2, request, MPI_STATUS_IGNORE);
    for (int i=0; i<N*N/size; i++)
      subB[i] = recv[i];
    tic = chrono::steady_clock::now();//time
    comm_time += chrono::duration<double>(tic - toc).count();//communication time?
  }
  MPI_Allgather(&subC[0], N*N/size, MPI_FLOAT, &C[0], N*N/size, MPI_FLOAT, MPI_COMM_WORLD);

  #pragma omp parallel for
  for (int i=0; i<N; i++)
    for (int k=0; k<N; k++)
      for (int j=0; j<N; j++)
        C[N*i+j] -= A[N*i+k] * B[N*k+j];
  double err = 0;
  for (int i=0; i<N; i++)
    for (int j=0; j<N; j++)
      err += fabs(C[N*i+j]);
  if(rank==0){
    double time = comp_time+comm_time;//total time
    printf("N    : %d\n",N);
    printf("comp : %lf s\n", comp_time);
    printf("comm : %lf s\n", comm_time);
    printf("total: %lf s (%lf GFlops)\n", time, 2.*N*N*N/time/1e9);
    printf("error: %lf\n",err/N/N);
  }
  cudaFree(A);
  cudaFree(B);
  cudaFree(C);
  cudaFree(subA);
  cudaFree(subB);
  cudaFree(subC);
  cudaFree(recv);
  MPI_Finalize();
}
