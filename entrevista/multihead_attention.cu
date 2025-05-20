/*
 * double* E;
 *
 * struct multihead_attention{
 *   double W_V, W_Q, W_K, W_O;
 *
 *   double* pass(double* E) {
 *     ...
 *     return multihead;
 *   }
 * }
 */ 

#include "stdio.h"
#include "cmath"

const int d_model = 4; 
const int n_heads = 2;
const int C = 2; // Janela de contexto
const int D = d_model / n_heads; // d_k 
const int block_size_x = 8, block_size_y = 8, block_size_z = 16;
const double sqrtD = sqrtl(D);

// funções auxiliares:
// Retorna double aleatório entre -1 e +1
double rand_double(){
  double min = -1, max = +1;
  double range = (max - min);
  return min + (double) rand() / (RAND_MAX / range);
}

// Teto da divisão de a por b
int ceil_div(int a, int b){
  return (a + b - 1) / b;
}

// device_W_x: (n_heads, D, d_model)
// device_E: (C, d_model) 
// device_x: (n_heads, C, D)
__global__ void projecao_linear(double *device_W_X, double *device_E, double *device_X, int C, int D, int d_model, int n_heads){
  int h = threadIdx.x + (blockDim.x * blockIdx.x); // head atual
  int i = threadIdx.y + (blockDim.y * blockIdx.y); // linha do device_E
  int j = threadIdx.z + (blockDim.z * blockIdx.z); // linha do device_W_X
  
  if (h < n_heads and i < C and j < D) {
    double soma = 0;
    for (int idx = 0; idx < d_model; ++idx) {
      soma += device_W_X[(h * D * d_model) + (j * d_model) + idx] * device_E[(i * d_model) + idx];
    }
    device_X[(h * C * D) + (i * D) + j] = soma;
  }
}

// k : (n_heads, C, D)
// K_t : (n_heads, D, C)
__global__ void transpor(double* device_K, double *device_K_transposto, int C, int D, int n_heads){
  int h = threadIdx.x + (blockDim.x * blockIdx.x); 
  int i = threadIdx.y + (blockDim.y * blockIdx.y); 
  int j = threadIdx.z + (blockDim.z * blockIdx.z); 

  if (h < n_heads and i < C and j < D) {
    device_K_transposto[(h * D * C) + (j * C) + i] = device_K[(h * C * D) + (i * D) + j];
  } 
}

// Q: (n_heads, C, D)
// K^T: (n_heads, D, C)
// device_H: (n_heads, C, C)
__global__ void produto_interno(double *Q, double *K_transposto, double *device_H, int C, int D, int n_heads, double sqrtD) {
  int h = threadIdx.x + (blockDim.x * blockIdx.x);
  int j = threadIdx.y + (blockDim.y * blockIdx.y);
  int i = threadIdx.z + (blockDim.z * blockIdx.z);

  if (h < n_heads and j < C and i < C) {
    if (j <= i) {
      double soma = 0;
      for (int idx = 0; idx < D; ++idx) {
        soma += Q[(h * C * D) + (i * D) + idx] * K_transposto[(h * D * C) + (idx * C) + j];
      }
      device_H[(h * C * C) + (i * C) + j] = soma / sqrtD;
    } else {
      device_H[(h * C * C) + (i * C) + j] = -INFINITY;
    }
  }
}

// device_HL: (n_heads, C, C)
__global__ void softmax(double *device_H, int C, int n_heads){
  int h = threadIdx.x + (blockDim.x * blockIdx.x);
  int i = threadIdx.y + (blockDim.y * blockIdx.y);
  
  if (h < n_heads and i < C) {
    double soma = 0;
    for (int idx = 0; idx < C; ++idx) {
      if (device_H[(h * C * C) + (i * C) + idx] != -INFINITY) {
        soma += exp(device_H[(h * C * C) + (i * C) + idx]);
      }
    }
    for (int idx = 0; idx < C; ++idx) {
      if (device_H[(h * C * C) + (i * C) + idx] != -INFINITY) {
        device_H[(h * C * C) + (i * C) + idx] = exp(device_H[(h * C * C) + (i * C) + idx]) / soma;
      } else {
        device_H[(h * C * C) + (i * C) + idx] = 0;
      }
    }
  }
}

// device_H: (n_heads, C, C)
// device_V: (n_heads, C, D)
// device_A: (n_heads, C, D)
__global__ void matmul(double *device_H, double *device_V, double *device_A, int C, int D, int n_heads){
  int h = threadIdx.x + (blockDim.x * blockIdx.x); // head atual
  int j = threadIdx.y + (blockDim.y * blockIdx.y); // coluna de V
  int i = threadIdx.z + (blockDim.z * blockIdx.z); // linha de H
                                                 
  if (h < n_heads and j < D and i < C) {
    double soma = 0;
    for (int idx = 0; idx < C; ++idx) {
      soma += device_H[(h * C * C) + (i * C) + idx] * device_V[(h * C * D) + (idx * D) + j];
    }
    device_A[(h * C * D) + (i * D) + j] = soma;
  }                                                 
}

// device_A: (n_heads, C, D)
// deice_concat: (C, d_model)
__global__ void concat(double *device_A, double *device_concat, int C, int D, int n_heads){
  int h = threadIdx.x + (blockDim.x * blockIdx.x); // head atual
  int i = threadIdx.y + (blockDim.y * blockIdx.y); // linha de A
  int j = threadIdx.z + (blockDim.z * blockIdx.z); // coluna de A
                                                   
  if (h < n_heads and i < C and j < D) {
    device_concat[(i * D * n_heads) + (h * D) + j] = device_A[(h * C * D) + (i * D) + j];
  }
}

// device_concat: (C, d_model)
// device_W_O: (d_model, d_model)
// multihead: (C, d_model)
__global__ void projecao_final(double *device_concat, double *device_W_O, double* multihead, int C, int d_model){
  int j = threadIdx.x + (blockIdx.x * blockDim.x);
  int i = threadIdx.y + (blockIdx.y * blockDim.y);
  
  if (j < d_model and i < C) {
    double soma = 0;
    for (int idx = 0; idx < d_model; ++idx) {
      soma += device_concat[(i * d_model) + idx] * device_W_O[(idx * d_model) + j];
    }
    multihead[(i * d_model) + j] = soma;
  }
}


struct multihead_attention{
  double *host_W_V, *host_W_Q, *host_W_K, *host_W_O;
  double *device_W_V, *device_W_Q, *device_W_K, *device_W_O;

  double *host_multihead;

  multihead_attention(){
    host_W_V = (double *) malloc(n_heads * D * d_model * sizeof(double));
    host_W_Q = (double *) malloc(n_heads * D * d_model * sizeof(double));
    host_W_K = (double *) malloc(n_heads * D * d_model * sizeof(double));
    host_W_O = (double *) malloc(d_model * d_model * sizeof(double));
    
    for (int i = 0; i < n_heads * D * d_model; ++i) {
      host_W_V[i] = rand_double();
      host_W_Q[i] = rand_double();
      host_W_K[i] = rand_double();
      host_W_O[i] = rand_double();
    }

    cudaMalloc(&device_W_V, n_heads * D * d_model * sizeof(double));
    cudaMalloc(&device_W_Q, n_heads * D * d_model * sizeof(double));
    cudaMalloc(&device_W_K, n_heads * D * d_model * sizeof(double));
    cudaMalloc(&device_W_O, d_model * d_model * sizeof(double));

    cudaMemcpy(device_W_V, host_W_V, n_heads * D * d_model * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(device_W_Q, host_W_Q, n_heads * D * d_model * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(device_W_K, host_W_K, n_heads * D * d_model * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(device_W_O, host_W_O, d_model * d_model * sizeof(double), cudaMemcpyHostToDevice);

    host_multihead = (double *) malloc(C * d_model * sizeof(double));
  }

  double* pass(double *E) {

    // Passando a matriz de embedding para a gpu
    double *device_E;
    cudaMalloc(&device_E, C * d_model * sizeof(double));
    cudaMemcpy(device_E, E, C * d_model * sizeof(double), cudaMemcpyHostToDevice);

    double *device_V, *device_Q, *device_K;
    cudaMalloc(&device_V, n_heads * C * D * sizeof(double));
    cudaMalloc(&device_Q, n_heads * C * D * sizeof(double));
    cudaMalloc(&device_K, n_heads * C * D * sizeof(double));

    // Cálculo das projeções lineares
    dim3 grid_dim_pl(ceil_div(n_heads, block_size_x), ceil_div(C, block_size_y), ceil_div(D, block_size_z));
    dim3 block_dim_pl(block_size_x, block_size_y, block_size_z);
    projecao_linear<<<grid_dim_pl, block_dim_pl>>>(device_W_V, device_E, device_V, C, D, d_model, n_heads);
    projecao_linear<<<grid_dim_pl, block_dim_pl>>>(device_W_Q, device_E, device_Q, C, D, d_model, n_heads);
    projecao_linear<<<grid_dim_pl, block_dim_pl>>>(device_W_K, device_E, device_K, C, D, d_model, n_heads);
    cudaDeviceSynchronize();

    double *host_V = (double *) malloc(n_heads * C * D * sizeof(double));
    cudaMemcpy(host_V, device_V, n_heads * C * D * sizeof(double), cudaMemcpyDeviceToHost);

    printf("V:\n");
    for (int i = 0; i < n_heads; ++i) {
      printf("Head %d:\n", i);
      for (int j = 0; j < C; ++j) {
        for (int k = 0; k < D; ++k) {
          printf("%lf ", host_V[(i * C * D) + (j * D) + k]);
        } 
        printf("\n");
      }
    }
    printf("\n");

    double *host_Q = (double *) malloc(n_heads * C * D * sizeof(double));
    cudaMemcpy(host_Q, device_Q, n_heads * C * D * sizeof(double), cudaMemcpyDeviceToHost);
    printf("Q:\n");
    for (int i = 0; i < n_heads; ++i) {
      printf("Head %d:\n", i);
      for (int j = 0; j < C; ++j) {
        for (int k = 0; k < D; ++k) {
          printf("%lf ", host_Q[(i * C * D) + (j * D) + k]);
        } 
        printf("\n");
      }
    }
    printf("\n");

    double *host_K = (double *) malloc(n_heads * C * D * sizeof(double));
    cudaMemcpy(host_K, device_K, n_heads * C * D * sizeof(double), cudaMemcpyDeviceToHost);
    printf("K:\n");
    for (int i = 0; i < n_heads; ++i) {
      printf("Head %d:\n", i);
      for (int j = 0; j < C; ++j) {
        for (int k = 0; k < D; ++k) {
          printf("%lf ", host_K[(i * C * D) + (j * D) + k]);
        } 
        printf("\n");
      }
    }
    printf("\n");

    // Transpõe a matriz K
    double *device_K_transposto;
    cudaMalloc(&device_K_transposto, n_heads * D * C * sizeof(double));
    dim3 grid_dim_t = grid_dim_pl;
    dim3 block_dim_t = block_dim_pl;
    transpor<<<grid_dim_t, block_dim_t>>>(device_K, device_K_transposto, C, D, n_heads);
    cudaDeviceSynchronize();
    
    double *host_K_transposto = (double *) malloc(n_heads * D * C * sizeof(double));
    cudaMemcpy(host_K_transposto, device_K_transposto, n_heads * D * C * sizeof(double), cudaMemcpyDeviceToHost);
    printf("K transposto:\n");
    for (int h = 0; h < n_heads; ++h) {
      printf("Head %d:\n", h);
      for (int i = 0; i < D; ++i) {
        for (int j = 0; j < C; ++j) {
          printf("%lf ", host_K_transposto[(h * D * C) + (i * C) + j]);
        }
        printf("\n");
      }
    }
    printf("\n");
    
    // Multiplicação Q por K^T
    double *device_H;
    cudaMalloc(&device_H, n_heads * C * C * sizeof(double));
    dim3 grid_dim_pi(ceil_div(n_heads, block_size_x), ceil_div(C, block_size_y), ceil_div(C, block_size_z));
    dim3 block_dim_pi = block_dim_pl;
    produto_interno<<<grid_dim_pi, block_dim_pi>>>(device_Q, device_K_transposto, device_H, C, D, n_heads, sqrtD);
    cudaDeviceSynchronize();

    double *host_H = (double *) malloc(n_heads * C * C * sizeof(double));
    cudaMemcpy(host_H, device_H, n_heads * C * C * sizeof(double), cudaMemcpyDeviceToHost);
    printf("Device H\n");
    for (int h = 0; h < n_heads; ++h) {
      for (int i = 0; i < C; ++i) {
        for (int j = 0; j < C; ++j) {
          printf("%lf ", host_H[(h * C * C) + (i * C) + j]);
        }
        printf("\n");
      }
    }
    printf("\n");

    // Aplica softmax em H
    dim3 grid_dim_s(ceil_div(n_heads, block_size_x), ceil_div(C, block_size_y));
    dim3 block_dim_s(block_size_x, block_size_y);
    softmax<<<grid_dim_s, block_dim_s>>>(device_H, C, n_heads);
    cudaDeviceSynchronize();
    
    cudaMemcpy(host_H, device_H, n_heads * C * C * sizeof(double), cudaMemcpyDeviceToHost);
    printf("Softmax:\n");
    for (int h = 0; h < n_heads; ++h) {
      for (int i = 0; i < C; ++i) {
        for (int j = 0; j < C; ++j) {
          printf("%lf ", host_H[(h * C * C) + (i * C) + j]);
        }
        printf("\n");
      }
    }
    printf("\n");

    // Obtém Attention em cada head 
    double *device_A;
    cudaMalloc(&device_A, n_heads * C * D * sizeof(double));
    dim3 grid_dim_mm(ceil_div(n_heads, block_size_x), ceil_div(D, block_size_y), ceil_div(C, block_size_z));
    dim3 block_dim_mm = block_dim_pl;
    matmul<<<grid_dim_mm, block_dim_mm>>>(device_H, device_V, device_A, C, D, n_heads);
    cudaDeviceSynchronize();

    double *host_A = (double *) malloc(n_heads * C * D * sizeof(double));
    cudaMemcpy(host_A, device_A, n_heads * C * D * sizeof(double), cudaMemcpyDeviceToHost);
    printf("Device A:\n");
    for (int h = 0; h < n_heads; ++h) {
      for (int i = 0; i < C; ++i) {
        for (int j = 0; j < D; ++j) {
          printf("%lf ", host_A[(h * C * D) + (i * D) + j]);
        }
        printf("\n");
      }
    }
    printf("\n");

    // Concatena tudo em concat
    double *device_concat;
    cudaMalloc(&device_concat, C * d_model * sizeof(double));
    dim3 grid_dim_c = grid_dim_pl;
    dim3 block_dim_c = block_dim_pl;
    concat<<<grid_dim_c, block_dim_c>>> (device_A, device_concat, C, D, n_heads);
    cudaDeviceSynchronize();

    double *host_concat = (double *) malloc(C * d_model * sizeof(double));
    cudaMemcpy(host_concat, device_concat, C * d_model * sizeof(double), cudaMemcpyDeviceToHost);
      
    printf("Concat:\n");
    for (int i = 0; i < C; ++i) {
      for (int j = 0; j < d_model; ++j) {
        printf("%lf ", host_concat[(i * d_model) + j]);
      }
      printf("\n");
    }
    printf("\n");

    // Calcula o resultado final de multihead
    double *multihead; 
    cudaMalloc(&multihead, C * d_model * sizeof(double));
    dim3 grid_dim_pf(ceil_div(d_model, block_size_x), ceil_div(C, block_size_y));
    dim3 block_dim_pf = block_dim_s;
    projecao_final<<<grid_dim_pf, block_dim_pf>>>(device_concat, device_W_O, multihead, C, d_model);
    cudaDeviceSynchronize();

    cudaMemcpy(host_multihead, multihead, C * d_model * sizeof(double), cudaMemcpyDeviceToHost);
    
    return host_multihead;
  }
};

int main(){
  srand(998244353);

  double *E = (double *) malloc(C * d_model * sizeof(double));
  for (int i = 0; i < C * d_model; ++i) {
    E[i] = rand_double();
  }

  multihead_attention mha;
  double *res =  mha.pass(E);

  printf("Resultado do attention:\n");

  for (int i = 0; i < C; ++i) {
    for (int j = 0; j < d_model; ++j) {
      printf("%lf ", res[(i * d_model) + j]);
    }
    printf("\n");
  }
}

