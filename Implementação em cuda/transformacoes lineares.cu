// Implementação da aplicação de camadas lineares
// W_V, W_Q e W_K na matriz de embedding

#include "stdio.h"

// constantes gpt-3
// const int d_model = 12288;
// const int num_heads = 96;
// const int C = 2048; // Tamanho do context window
// const int vocabulary_size = 50257;
// const int D = d_model / num_heads;
// const int thread_size = 32;

// constantes meu modelo
const int d_model = 4;
const int num_heads = 2;
const int C = 2; // Tamanho do context window
const int D = d_model / num_heads; // = 2
const int thread_size = 32;

// Teto da divisão de a por b
int ceil_div(int a, int b) {
  return (a + b - 1) / b;
}

// Pega um número aleatório entre -1 e 1
double rand_double() {
  double min = -1, max = 1;
  double range = (max - min);
  return min + (double) rand() / (RAND_MAX / range);
}


// Aplica uma camada linear A de forma (k, m) a uma matriz B de forma (n, m)
// para obter uma matriz resultante C de forma (n, k)
__global__ void transformacao_linear(double *A, double *B, double *C, int n, int m, int k) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  int j = threadIdx.y + blockDim.y * blockIdx.y;
  
  // i é o índice do vetor que vamos aplicar a camada agora
  // j é qual dimensão da camada estamos agora para aplicar a transformação
  if (i < n and j < k) {
    double soma = 0;
    for (int idx = 0; idx < m; ++idx) {
      soma += A[j * m + idx] * B[i * m + idx];
    }
    C[i * k + j] = soma;
  }
}

// Usado para verificar a validade da aplicação da camada linear usando a GPU 
void transformacao_linear_cpu(double *A, double *B, double *C, int n, int m, int k) {
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < k; ++j) {
      double soma = 0;
      for (int idx = 0; idx < m; ++idx) {
        soma += A[j * m + idx] * B[i * m + idx];
      }
      C[i * k + j] = soma;
    }
  }
}


struct self_attention_head{
  double *host_W_V, *host_W_Q, *host_W_K;
  double *device_W_V, *device_W_Q, *device_W_K, *device_V, *device_Q, *device_K;
  double *device_E;
    
  self_attention_head(){
    // As seguintes são camadas lineares que transformam a
    // matriz de embedding da forma (C, d_model) para (C, D)
    host_W_V = (double *) malloc(D * d_model * sizeof(double));
    host_W_Q = (double *) malloc(D * d_model * sizeof(double));
    host_W_K = (double *) malloc(D * d_model * sizeof(double));

    // Define inicialmente valores aleatórios para cada 
    // valor em W_V, W_Q, e W_K
    for (int i = 0; i < D * d_model; ++i) {
      host_W_V[i] = rand_double();
      host_W_Q[i] = rand_double();
      host_W_K[i] = rand_double();
    }

    // Colocando essas matrizes geradas na GPU
    cudaMalloc(&device_W_V, D * d_model * sizeof(double));
    cudaMalloc(&device_W_Q, D * d_model * sizeof(double));
    cudaMalloc(&device_W_K, D * d_model * sizeof(double));
    cudaMemcpy(device_W_V, host_W_V, D * d_model * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(device_W_Q, host_W_Q, D * d_model * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(device_W_K, host_W_K, D * d_model * sizeof(double), cudaMemcpyHostToDevice);

    // Alocando espaço para as matrizes V, Q, K. 
    // os resultados das projeções calculadas
    cudaMalloc(&device_V, C * D * sizeof(double));
    cudaMalloc(&device_Q, C * D * sizeof(double));
    cudaMalloc(&device_K, C * D * sizeof(double));

    // Aloca espaço para a matriz de embedding na GPU
    cudaMalloc(&device_E, C * d_model * sizeof(double));
  }

  void pass_embedding(double *E){
    // E é o embedding do input. Ele tem forma (C, d_model),
    // onde C é o tamannho da janela de contexto e d_model é 
    // a dimensão do modelo. Primeiro, temos que fazer uma 
    // transformação linear para transformar E de (C, d_model)
    // para (C, D), para termos exatamente C * d_model valores 
    // nos embeddings entre todas as heads. 

    // Copia E para a GPU
    cudaMemcpy(device_E, E, C * d_model * sizeof(double), cudaMemcpyHostToDevice);
    
    // Faz as transformações lineares na GPU
    dim3 grid_dim_tl(ceil_div(C, thread_size), ceil_div(D, thread_size));
    dim3 block_dim_tl(thread_size, thread_size);
    transformacao_linear<<<grid_dim_tl, block_dim_tl>>>(device_W_V, device_E, device_V, D, d_model, C);
    transformacao_linear<<<grid_dim_tl, block_dim_tl>>>(device_W_Q, device_E, device_Q, D, d_model, C);
    transformacao_linear<<<grid_dim_tl, block_dim_tl>>>(device_W_K, device_E, device_K, D, d_model, C);
    cudaDeviceSynchronize();

    // --- testando validade da transformação linear ---
    // Faz as transformações lineares na CPU
    double *host_V, *host_Q, *host_K;
    host_V = (double *) malloc(C * D * sizeof(double));
    host_Q = (double *) malloc(C * D * sizeof(double));
    host_K = (double *) malloc(C * D * sizeof(double));
    transformacao_linear_cpu(host_W_V, E, host_V, D, d_model, C);
    transformacao_linear_cpu(host_W_Q, E, host_Q, D, d_model, C);
    transformacao_linear_cpu(host_W_K, E, host_K, D, d_model, C);

    // Cria cópia das matrizes obtidas na gpu 
    double *device_copy_V = (double *) malloc(C * D * sizeof(double));
    double *device_copy_Q = (double *) malloc(C * D * sizeof(double));
    double *device_copy_K = (double *) malloc(C * D * sizeof(double));
    cudaMemcpy(device_copy_V, device_V, C * D * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(device_copy_Q, device_Q, C * D * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(device_copy_K, device_K, C * D * sizeof(double), cudaMemcpyDeviceToHost);

    // Faz a verificação
    for (int i = 0; i < C; ++i) {
      for (int j = 0; j < D; ++j) {
        if (fabs(device_copy_V[i * D + j] - host_V[i * D + j]) > 1e-9) {
          printf("Deu errado na matriz de valores (V):\n");
          printf("i: %d j: %d\n", i, j);
          printf("Valor na CPU: %lf\n", host_V[i * D + j]);
          printf("Valor na GPU: %lf\n", device_copy_V[i * D + j]);
          return;
        }
        if (fabs(device_copy_Q[i * D + j] - host_Q[i * D + j]) > 1e-9) {
          printf("Deu errado na matriz de consultas (Q):\n");
          printf("i: %d j: %d\n", i, j);
          printf("Valor na CPU: %lf\n", host_Q[i * D + j]);
          printf("Valor na GPU: %lf\n", device_copy_Q[i * D + j]);
          return;
        }
        if (fabs(device_copy_K[i * D + j] - host_K[i * D + j]) > 1e-9) {
          printf("Deu errado na matriz de chaves (K):\n");
          printf("i: %d j: %d\n", i, j);
          printf("Valor na CPU: %lf\n", host_K[i * D + j]);
          printf("Valor na GPU: %lf\n", device_copy_K[i * D + j]);
          return;
        }
      }
    }
    printf("Deu certo!\n");
    printf("Matriz de valores (V):\n");
    for (int i = 0; i < C; ++i) {
      for (int j = 0; j < D; ++j) {
        printf("%lf ", host_V[i * D + j]);
      }
      printf("\n");
    }
    printf("Matriz de consultas (Q):\n");
    for (int i = 0; i < C; ++i) {
      for (int j = 0; j < D; ++j) {
        printf("%lf ", host_Q[i * D + j]);
      }
      printf("\n");
    }
    printf("Matriz de chaves (K):\n");
    for (int i = 0; i < C; ++i) {
      for (int j = 0; j < D; ++j) {
        printf("%lf ", host_K[i * D + j]);
      }
      printf("\n");
    }
    // --- Fim do teste da transformação linear--
  }
};


int main(){
  srand(1337);

  double *E = (double *) malloc(C * d_model * sizeof(double));
  for (int i = 0; i < C * d_model; ++i) {
    E[i] = rand_double();
  }
  
  self_attention_head sa;
  sa.pass_embedding(E);
}
