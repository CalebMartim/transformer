//#include "stdio.h"
#include "cmath"

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
const int D = d_model / num_heads;
const int thread_size = 32;
const double sqrtD = sqrtl(D); 

// Teto da divisão de a por b
int ceil_div(int a, int b) {
  return (a + b - 1) / b;
}

// Aplica uma camada linear A de forma (k, m) à uma matriz B de forma (n, m)
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

// Transpõe uma matriz A de forma (n, m) e coloca o resultado 
// em B, que tem forma (m, n)
__global__ void transpor(double *A, double *B, int n, int m) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  int j = threadIdx.y + blockDim.y * blockIdx.y;

  if (i < n and j < m) {
    B[j * n + i] = A[i * m + j];
  }
}

void transpor_cpu(double *A, double *B, int n, int m) {
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < m; ++j) {
      B[j * n + i] = A[i * m + j];
    }
  }
}

// Multiplica uma matiz A de forma (n, k) com uma matriz B de forma (k, m) 
// e coloca o resultado na matriz C, que tem forma (n, m)
__global__ void multiplica_matrizes(double *A, double *B, double *C, int n, int m, int k) {
  int j = threadIdx.x + blockDim.x * blockDim.x;
  int i = threadIdx.y + blockDim.y * blockIdx.y;

  if (j < m and i < n) {
    // No caso de masked attention é necessário esse if
    if (j <= i) {
      double soma = 0;
      for (int idx = 0; idx < k; ++idx) {
        soma += A[i * k + idx] * B[idx * m + j];
      }
      C[i * m + j] = soma;

      // No bloco de self attention:
      C[i * m + j] = soma / sqrtD;
    } else {
      C[i * m + j] = -INFINITY; 
    }
  }
}


// S(A) = (n, k), S(B) = (k, m), S(C) = (n, m)
void multiplicar_cpu(double *A, double *B, double *C, int n, int m, int k) {
  for (int j = 0; j < m; ++j) {
    for (int i = 0; i < n; ++i) {
      double soma = 0;
      for (int idx = 0; idx < k; ++idx) {
        soma += A[i * k + idx] * B[idx * m + j];
      }
      C[i * m + j] = soma; 
    }
  }
}

// Pega uma matriz A de forma (n, m) e aplica a função softmax
// em cada uma de suas linhas
__global__ void softmax(double *A, int n, int m){
  int i = threadIdx.x + blockDim.x * blockIdx.x;

  if (i < n) {
    double soma = 0;
    for (int idx = 0; idx < m; ++idx) {
      soma += exp(A[i * m + idx]);
    }
    for (int idx = 0; idx < m; ++idx) {
      A[i * m + idx] = exp(A[i * m + idx]) / soma;
    }
  }
}

void softmax_cpu(double *A, int n, int m) {
  for (int i = 0; i < n; ++i) {
    double soma = 0;
    for (int idx = 0; idx < m; ++idx) {
      soma += exp(A[i * m + idx]);
    }
    for (int idx = 0; idx < m; ++idx) {
      A[i * m + idx] = exp(A[i * m + idx]) / soma;
    }
  }
}


struct self_attention_head{
  double *host_W_V, *host_W_Q, *host_W_K;
  double *device_W_V, *device_W_Q, *device_W_K, 
         *device_V, *device_Q, *device_K,
         *device_K_transposto;
  double *device_E, *device_H;
    
  self_attention_head(){
    // As seguintes são camadas lineares que transformam a
    // matriz de embedding da forma (C, d_model) para (C, D)
    host_W_V = (double *) malloc(D * d_model * sizeof(double));
    host_W_Q = (double *) malloc(D * d_model * sizeof(double));
    host_W_K = (double *) malloc(D * d_model * sizeof(double));

    // Define inicialmente valores aleatórios para cada 
    // valor em W_V, W_Q, e W_K
    for (int i = 0; i < D * d_model; ++i) {
      host_W_V[i] = (double) rand() / RAND_MAX; 
      host_W_Q[i] = (double) rand() / RAND_MAX; 
      host_W_K[i] = (double) rand() / RAND_MAX; 

      // Valores menores para verificação
      host_W_V[i] = rand() % 10;
      host_W_Q[i] = rand() % 10;
      host_W_K[i] = rand() % 10;
    }

    // Colocando essas matrizes na GPU
    cudaMalloc(&device_W_V, D * d_model * sizeof(double));
    cudaMalloc(&device_W_Q, D * d_model * sizeof(double));
    cudaMalloc(&device_W_K, D * d_model * sizeof(double));
    cudaMemcpy(device_W_V, host_W_V, D * d_model * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(device_W_Q, host_W_Q, D * d_model * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(device_W_K, host_W_K, D * d_model * sizeof(double), cudaMemcpyHostToDevice);

    // Preparando as matrizes V, Q, K para colocarmos 
    // nossos valores depois de passar 
    // a matriz de embedding pelas camadas lineares
    cudaMalloc(&device_V, C * D * sizeof(double));
    cudaMalloc(&device_Q, C * D * sizeof(double));
    cudaMalloc(&device_K, C * D * sizeof(double));

    // Auxiliares
    // Prepara a matriz de embedding na GPU
    cudaMalloc(&device_E, C * d_model * sizeof(double));

    // Prepara a matriz transposta de K na GPU
    cudaMalloc(&device_K_transposto, D * C * sizeof(double));

    // Matriz auxiliar para fazer a multiplicação entre Q e K^T
    cudaMalloc(&device_H, C * C * sizeof(double));
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
    dim3 grid_dim(ceil_div(C, thread_size), ceil_div(D, thread_size));
    dim3 block_dim(thread_size, thread_size);
    transformacao_linear<<<grid_dim, block_dim>>>(device_W_V, device_E, device_V, D, d_model, C);
    transformacao_linear<<<grid_dim, block_dim>>>(device_W_Q, device_E, device_Q, D, d_model, C);
    transformacao_linear<<<grid_dim, block_dim>>>(device_W_K, device_E, device_K, D, d_model, C);
    cudaDeviceSynchronize();
    
    // Vamos transpor o vetor de keys e multiplicar Q * K^T 
    transpor<<<grid_dim, block_dim>>>(device_K, device_K_transposto, C, D);

    dim3 grid_dim_multiplicacao(ceil_div(C, thread_size), ceil_div(C, thread_size));
    multiplica_matrizes<<<grid_dim_multiplicacao, block_dim>>>(device_Q, device_K_transposto, device_H, C, C, D);

    int grid_dim_softmax = ceil_div(C, thread_size);
    
    softmax<<<grid_dim_softmax, thread_size>>>(device_H, C, C);

    // Testando na cpu
    double *host_Q = (double *) malloc(C * D * sizeof(double)), 
           *host_K = (double *) malloc(C * D * sizeof(double)),
           *host_K_transposto = (double *) malloc(D * C * sizeof(double)),
           *host_H = (double *) malloc(C * C * sizeof(double));

    cudaMemcpy(host_Q, device_Q, C * D * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_K, device_K, C * D * sizeof(double), cudaMemcpyDeviceToHost);

    transpor_cpu(device_K, device_K_transposto, C, D);
    
    multiplicar_cpu(host_Q, host_K_transposto, host_H, C, C, D);

    softmax_cpu(host_H, C, C);
  }
};


int main(){
  srand(1337);

  double *E = (double *) malloc(C * d_model * sizeof(double));
  for (int i = 0; i < C * d_model; ++i) {
    E[i] = (double) rand() / RAND_MAX;

    // Valor menor para teste
    E[i] = rand() % 10;
  }
  
  self_attention_head sa;
  sa.pass_embedding(E);
}
