// Após calcular H = softmax(Q(K^T) / sqrt(D)),
// multiplicamos H por V, finalizando o 
// cálculo da função de atenção

#include "stdio.h"
#include "cmath"

// constantes no modelo gpt-3, para referência
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
const double sqrtD = sqrtl(D); 

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

// Transpõe uma matriz A de forma (n, m) e coloca o resultado 
// em B, que tem forma (m, n)
__global__ void transpor(double *A, double *B, int n, int m) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  int j = threadIdx.y + blockDim.y * blockIdx.y;

  if (i < n and j < m) {
    B[j * n + i] = A[i * m + j];
  }
}

// Multiplica uma matriz A de forma (n, k) com uma matriz B de forma (k, m) 
// e coloca o resultado na matriz C, que tem forma (n, m)
// Nesta função, aplicamos o conceito de masked self attention,
// para que nenhum token obtenha informações sobre tokens em posições à frente
// e dividimos todo valor por D por questões de normalização de valores
__global__ void primeira_multiplicacao(double *A, double *B, double *C, int n, int m, int k, double sqrtD) {
  int j = threadIdx.x + blockDim.x * blockIdx.x;
  int i = threadIdx.y + blockDim.y * blockIdx.y;

  if (j < m and i < n) {
    // No caso de masked attention, só calculamos o produto interno 
    // Q_i * K_j quando j <= i, para prevenir "spoilers" pro modelo
    if (j <= i) {
      double soma = 0;
      for (int idx = 0; idx < k; ++idx) {
        soma += A[i * k + idx] * B[idx * m + j];
      }
      C[i * m + j] = soma / sqrtD; // A divisão normaliza o resultado 
    } else {
      C[i * m + j] = -INFINITY; 
    }
  }
}

// Multiplica uma matriz A de forma (n, k) com uma matriz B de forma (k, m) 
// e coloca o resultado na matriz C, que tem forma (n, m)
__global__ void segunda_multiplicacao(double *A, double *B, double *C, int n, int m, int k) {
  int j = threadIdx.x + blockDim.x * blockIdx.x;
  int i = threadIdx.y + blockDim.y * blockIdx.y;

  if (j < m and i < n) {
    double soma = 0;
    for (int idx = 0; idx < k; ++idx) {
      soma += A[i * k + idx] * B[idx * m + j];
    }
    C[i * m + j] = soma;
  }
}

void segunda_multiplicacao_cpu(double *A, double *B, double *C, int n, int m, int k) {
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
      if (A[i * m + idx] != -INFINITY) { // caso contrário, ele não vai conseguir calcular a exponencial 
        soma += exp(A[i * m + idx]);
      }
    }
    for (int idx = 0; idx < m; ++idx) {
      if (A[i * m + idx] != -INFINITY) {
        A[i * m + idx] = exp(A[i * m + idx]) / soma;
      } else {
        A[i * m + idx] = 0;
      }
    }
  }
}

struct self_attention_head{
  double *host_W_V, *host_W_Q, *host_W_K;
  double *device_W_V, *device_W_Q, *device_W_K, 
         *device_V, *device_Q, *device_K,
         *device_K_transposto;
  double *device_E, *device_H, *device_A;
    
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

    // Colocando essas matrizes na GPU
    cudaMalloc(&device_W_V, D * d_model * sizeof(double));
    cudaMalloc(&device_W_Q, D * d_model * sizeof(double));
    cudaMalloc(&device_W_K, D * d_model * sizeof(double));
    cudaMemcpy(device_W_V, host_W_V, D * d_model * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(device_W_Q, host_W_Q, D * d_model * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(device_W_K, host_W_K, D * d_model * sizeof(double), cudaMemcpyHostToDevice);

    // Alocando espaço para as matrizes 
    // V, Q, K. Estas serão as matrizes que
    // resultam das projeções da matriz de 
    // embedding nas camadas lineares
    cudaMalloc(&device_V, C * D * sizeof(double));
    cudaMalloc(&device_Q, C * D * sizeof(double));
    cudaMalloc(&device_K, C * D * sizeof(double));

    // Aloca espaço para copiarmos a matriz de embedding para a GPU
    cudaMalloc(&device_E, C * d_model * sizeof(double));

    // Prepara a matriz transposta de K na GPU
    cudaMalloc(&device_K_transposto, D * C * sizeof(double));

    // Matriz auxiliar para fazer a multiplicação entre Q e K^T
    cudaMalloc(&device_H, C * C * sizeof(double));

    cudaMalloc(&device_A, C * D * sizeof(double));
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

    // Faz as transformações lineares paralelamente
    dim3 grid_dim_tl(ceil_div(C, thread_size), ceil_div(D, thread_size));
    dim3 block_dim_tl(thread_size, thread_size);
    transformacao_linear<<<grid_dim_tl, block_dim_tl>>>(device_W_V, device_E, device_V, D, d_model, C);
    transformacao_linear<<<grid_dim_tl, block_dim_tl>>>(device_W_Q, device_E, device_Q, D, d_model, C);
    transformacao_linear<<<grid_dim_tl, block_dim_tl>>>(device_W_K, device_E, device_K, D, d_model, C);
    cudaDeviceSynchronize();

    // Vamos transpor a matriz K para podermos fazer a multiplicação Q(K^T) 
    dim3 grid_dim_t = grid_dim_tl;
    dim3 block_dim_t = block_dim_tl;
    transpor<<<grid_dim_t, block_dim_t>>>(device_K, device_K_transposto, C, D);
    cudaDeviceSynchronize();
    
    // Fazemos a multiplicação, gerando uma matriz H de forma (C, C):
    dim3 grid_dim_pm(ceil_div(C, thread_size), ceil_div(C, thread_size));
    dim3 block_dim_pm = block_dim_tl;
    primeira_multiplicacao<<<grid_dim_pm, block_dim_pm>>>(device_Q, device_K_transposto, device_H, C, C, D, sqrtD);
    cudaDeviceSynchronize();

    // Aplicamos o softmax na matriz resultante da última multiplicação:
    int grid_dim_s = ceil_div(C, thread_size);
    int block_dim_s = thread_size;
    softmax<<<grid_dim_s, block_dim_s>>>(device_H, C, C);
    cudaDeviceSynchronize();
    
    // Multiplicamos a matriz H, que tem forma (C, C), com a matriz V
    // que tem forma (C, D) para finalizarmos o cálculo do attention
    dim3 grid_dim_sm(ceil_div(D, thread_size), ceil_div(C, thread_size));
    dim3 block_dim_sm(thread_size, thread_size);
    segunda_multiplicacao<<<grid_dim_sm, block_dim_sm>>>(device_H, device_V, device_A, C, D, C);

    // --- testando resultado da multiplicação ---
    double *host_H, *host_V, *host_A;
    host_H = (double *) malloc(C * C * sizeof(double));
    host_V = (double *) malloc(C * D * sizeof(double));
    host_A = (double *) malloc(C * D * sizeof(double));
    
    cudaMemcpy(host_H, device_H, C * C * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_V, device_V, C * D * sizeof(double), cudaMemcpyDeviceToHost);

    segunda_multiplicacao_cpu(host_H, host_V, host_A, C, D, C);

    double *device_copy_A = (double *) malloc(C * D * sizeof(double));
    cudaMemcpy(device_copy_A, device_A, C * D * sizeof(double), cudaMemcpyDeviceToHost);

    for (int i = 0; i < C; ++i) {
      for (int j = 0; j < D; ++j) {
        if (fabs(host_A[i * D + j] - device_copy_A[i * D + j]) > 1e-9) {
          printf("Deu errado\n");
          printf("i: %d j: %d\n", i, j);
          printf("Valor na CPU: %lf\n", host_A[i * D + j]);
          printf("Valor na GPU: %lf\n", device_copy_A[i * D + j]);
          return;
        }
      }
    } 
    printf("Deu certo!\n");
    printf("Resultado do attention:\n");
    for (int i = 0; i < C; ++i) {
      for (int j = 0; j < D; ++j) {
        printf("%lf ", host_A[i * D + j]);   
      }
      printf("\n");
    }
    // --- fim do teste ---
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
