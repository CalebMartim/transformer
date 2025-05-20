// Validando a transposição, produto interno e 
// aplicação do soft max em multi head

#include "stdio.h"
#include "cmath"

// constantes no modelo gpt-3, para referência
// const int d_model = 12288;
// const int n_heads = 96;
// const int C = 2048; // TamaHo do context window
// const int vocabulary_size = 50257;
// const int D = d_model / H;
// const int thread_size = 32;

// constantes meu modelo
const int d_model = 4;
const int n_heads = 2;
const int C = 2; // TamaHo do context window
const int D = d_model / n_heads; // = 2
const int block_dim_x = 8, block_dim_y = 8, block_dim_z = 16;
const double sqrtD = sqrtl(D); 

// Teto da divisão de a por b
int ceil_div(int a, int b) {
  return (a + b - 1) / b;
}

// Pega um número aleatório entre -1 e 1
double rand_double() {
  double min = -1, max = 1;
  double range = max - min;
  return (double) rand() / (RAND_MAX / range);
}

// Aplica uma camada linear A de forma (k, m) a uma matriz B de forma (n, m)
// para obter uma matriz resultante C de forma (n, k)
__global__ void transformacao_linear(double *A, double *B, double *C, int n, int m, int k, int n_heads) {
  int h = threadIdx.x + blockDim.x * blockIdx.x;
  int i = threadIdx.y + blockDim.y * blockIdx.y;
  int j = threadIdx.z + blockDim.z * blockIdx.z;
  
  // i é o índice do vetor que vamos aplicar a camada agora
  // j é qual dimensão da camada estamos agora para aplicar a transformação
  if (h < n_heads and i < n and j < k) {
    double soma = 0;
    for (int idx = 0; idx < m; ++idx) {
      soma += A[(h * k * m) + (j * m) + idx] * B[(i * m) + idx];
    }
    C[(h * n * k) + (i * k) + j] = soma;
  }
}

// Transpõe uma matriz A de forma (n, m) e coloca o resultado 
// em B, que tem forma (m, n)
__global__ void transpor(double *A, double *B, int n, int m, int n_heads) {
  int h = threadIdx.x + blockDim.x * blockIdx.x;
  int i = threadIdx.y + blockDim.y * blockIdx.y;
  int j = threadIdx.z + blockDim.z * blockIdx.z;

  if (h < n_heads and i < n and j < m) {
    B[(h * m * n) + (j * n) + i] = A[(h * n * m) + (i * m) + j];
  }
}

void transpor_cpu(double *K, double *K_transposto, int C, int D, int n_heads) {
  for (int h = 0; h < n_heads; ++h) {
    for (int i = 0; i < C; ++i) {
      for (int j = 0; j < D; ++j) {
        K_transposto[(h * D * C) + (j * C) + i] = K[(h * C * D) + (i * D) + j];
      }
    }
  }
}

// Multiplica uma matriz A de forma (n, k) com uma matriz B de forma (k, m) 
// e coloca o resultado na matriz C, que tem forma (n, m)
// Nesta função, aplicamos o conceito de masked self attention,
// para que neHum token obteHa informações sobre tokens em posições à frente
// e dividimos todo valor por D por questões de normalização de valores
__global__ void primeira_multiplicacao(double *A, double *B, double *C, int n, int m, int k, double sqrtD, int n_heads) {
  int h = threadIdx.x + blockDim.x * blockIdx.x;
  int j = threadIdx.y + blockDim.y * blockIdx.y;
  int i = threadIdx.z + blockDim.z * blockIdx.z;

  if (h < n_heads and j < m and i < n) {
    // No caso de masked attention, só calculamos o produto interno 
    // Q_i * K_j quando j <= i, para prevenir "spoilers" pro modelo
    if (j <= i) {
      double soma = 0;
      for (int idx = 0; idx < k; ++idx) {
        soma += A[(h * n * k) + (i * k) + idx] * B[(h * k * m) + (idx * m) + j];
      }
      C[(h * n * m) + (i * m) + j] = soma / sqrtD; // A divisão normaliza o resultado 
    } else {
      C[(h * n * m) + (i * m) + j] = -INFINITY; 
    }
  }
}

void produto_interno_escalado(double *Q, double *K_transposto, double *host_H, int C, int D, int n_heads) {
  for (int h = 0; h < n_heads; ++h) {
    for (int j = 0; j < C; ++j) {
      for (int i = 0; i < C; ++i) {
        if (j <= i) {
          double soma = 0;
          for (int idx = 0; idx < D; ++idx) {
            soma += Q[(h * C * D) + (i * D) + idx] * K_transposto[(h * D * C) + (idx * C) + j];
          }
          host_H[(h * C * C) + (i * C) + j] = soma / sqrtD;
        } else {
          host_H[(h * C * C) + (i * C) + j] = -INFINITY;
        }
      }
    }
  }
}

// Pega uma matriz A de forma (n, m) e aplica a função softmax
// em cada uma de suas liHas
__global__ void softmax(double *A, int n, int m, int n_heads){
  int h = threadIdx.x + blockDim.x * blockIdx.x;
  int i = threadIdx.y + blockDim.y * blockIdx.y;

  if (h < n_heads and i < n) {
    double soma = 0;
    for (int idx = 0; idx < m; ++idx) {
      if (A[(h * n * m) + (i * m) + idx] != -INFINITY) { // caso contrário, ele não vai conseguir calcular a exponencial 
        soma += exp(A[(h * n * m) + (i * m) + idx]);
      }
    }
    for (int idx = 0; idx < m; ++idx) {
      if (A[(h * n * m) + (i * m) + idx] != -INFINITY) {
        A[(h * n * m) + (i * m) + idx] = exp(A[(h * n * m) + (i * m) + idx]) / soma;
      } else {
        A[(h * n * m) + (i * m) + idx] = 0;
      }
    }
  }
}

void softmax_cpu(double *host_H, int C, int n_heads) {
  for (int h = 0; h < n_heads; ++h) {
    for (int i = 0; i < C; ++i) {
      double soma = 0;
      for (int j = 0; j < C; ++j) {
        if (host_H[(h * C * C) + (i * C) + j] != -INFINITY) {
          soma += exp(host_H[(h * C * C) + (i * C) + j]);
        }
      }
      for (int j = 0; j < C; ++j) {
        if (host_H[(h * C * C) + (i * C) + j] == -INFINITY) {
          host_H[(h * C * C) + (i * C) + j] = 0;
        } else {
          host_H[(h * C * C) + (i * C) + j] = exp(host_H[(h * C * C) + (i * C) + j]) / soma;
        }
      }
    }
  }
}

struct MultiHeadAttention{
  double *host_W_V, *host_W_Q, *host_W_K;
  double *device_W_V, *device_W_Q, *device_W_K, 
         *device_V, *device_Q, *device_K,
         *device_K_transposto;
  double *device_E, *device_H;

  MultiHeadAttention(){
    // As seguintes são camadas lineares que transformam a
    // matriz de embedding da forma (C, d_model) para (C, D)
    host_W_V = (double *) malloc(n_heads * D * d_model * sizeof(double));
    host_W_Q = (double *) malloc(n_heads * D * d_model * sizeof(double));
    host_W_K = (double *) malloc(n_heads * D * d_model * sizeof(double));

    // Define inicialmente valores aleatórios para cada 
    // valor em W_V, W_Q, e W_K
    for (int i = 0; i < n_heads * D * d_model; ++i) {
      host_W_V[i] = rand_double();
      host_W_Q[i] = rand_double();
      host_W_K[i] = rand_double();
    }

    // Colocando essas matrizes na GPU
    cudaMalloc(&device_W_V, n_heads * D * d_model * sizeof(double));
    cudaMalloc(&device_W_Q, n_heads * D * d_model * sizeof(double));
    cudaMalloc(&device_W_K, n_heads * D * d_model * sizeof(double));
    cudaMemcpy(device_W_V, host_W_V, n_heads * D * d_model * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(device_W_Q, host_W_Q, n_heads * D * d_model * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(device_W_K, host_W_K, n_heads * D * d_model * sizeof(double), cudaMemcpyHostToDevice);

    // Alocando espaço para as matrizes 
    // V, Q, K. Estas serão as matrizes que
    // resultam das projeções da matriz de 
    // embedding nas camadas lineares
    cudaMalloc(&device_V, n_heads * C * D * sizeof(double));
    cudaMalloc(&device_Q, n_heads * C * D * sizeof(double));
    cudaMalloc(&device_K, n_heads * C * D * sizeof(double));

    // Aloca espaço para copiarmos a matriz de embedding para a GPU
    // (não precisa multiplicar por H porque é único através das heads)
    cudaMalloc(&device_E, C * d_model * sizeof(double)); 

    // Prepara a matriz transposta de K na GPU
    cudaMalloc(&device_K_transposto, n_heads * D * C * sizeof(double));

    // Matriz auxiliar para fazer a multiplicação entre Q e K^T
    cudaMalloc(&device_H, n_heads * C * C * sizeof(double));
  }

  void pass_embedding(double *E){
    // E é o embedding do input. Ele tem forma (C, d_model),
    // onde C é o tamanHo da janela de contexto e d_model é 
    // a dimensão do modelo. Primeiro, temos que fazer uma 
    // transformação linear para transformar E de (C, d_model)
    // para (C, D), para termos exatamente C * d_model valores 
    // nos embeddings entre todas as heads. 

    // Copia E para a GPU
    cudaMemcpy(device_E, E, C * d_model * sizeof(double), cudaMemcpyHostToDevice);

    // Faz as transformações lineares paralelamente
    dim3 grid_dim_tl(ceil_div(n_heads, block_dim_x), ceil_div(C, block_dim_y), ceil_div(D, block_dim_z));
    dim3 block_dim_tl(block_dim_x, block_dim_y, block_dim_z);
    transformacao_linear<<<grid_dim_tl, block_dim_tl>>>(device_W_V, device_E, device_V, C, d_model, D, n_heads);
    transformacao_linear<<<grid_dim_tl, block_dim_tl>>>(device_W_Q, device_E, device_Q, C, d_model, D, n_heads);
    transformacao_linear<<<grid_dim_tl, block_dim_tl>>>(device_W_K, device_E, device_K, C, d_model, D, n_heads);
    cudaDeviceSynchronize();

    // copiando resultados na CPU:
    double *host_Q, *host_K;
    host_Q = (double *) malloc(n_heads * C * D * sizeof(double));
    host_K = (double *) malloc(n_heads * C * D * sizeof(double));
    cudaMemcpy(host_Q, device_Q, n_heads * C * D * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_K, device_K, n_heads * C * D * sizeof(double), cudaMemcpyDeviceToHost);

    // Vamos transpor a matriz K para podermos fazer a multiplicação Q(K^T) 
    dim3 grid_dim_t = grid_dim_tl;
    dim3 block_dim_t = block_dim_tl;
    transpor<<<grid_dim_t, block_dim_t>>>(device_K, device_K_transposto, C, D, n_heads);
    cudaDeviceSynchronize();

    // --- Olhando transposição ---
    // Fazendo a transposição na CPU
    double *host_K_transposto;
    host_K_transposto = (double *) malloc(n_heads * D * C * sizeof(double));
    transpor_cpu(host_K, host_K_transposto, C, D, n_heads);

    // Cópia da transposição da GPU para a CPU:
    double *device_copy_K_transposto = (double *) malloc(n_heads * D * C * sizeof(double));
    cudaMemcpy(device_copy_K_transposto, device_K_transposto, n_heads * D * C * sizeof(double), cudaMemcpyDeviceToHost);

    printf("Matrizes K transpostas GPU:\n");
    for (int h = 0; h < n_heads; ++h) {
      printf("head: %d\n", h);
      for (int i = 0; i < D; ++i) {
        for (int j = 0; j < C; ++j){
          printf("%lf ", device_copy_K_transposto[(h * D * C) + (i * C) + j]);
        }
        printf("\n");
      }
    }
    printf("matrizes K transpostas (CPU):\n");
    for (int h = 0; h < n_heads; ++h) {
      printf("head: %d\n", h);
      for (int i = 0; i < D; ++i) {
        for (int j = 0; j < C; ++j) {
          printf("%lf ", host_K_transposto[(h * D * C) + (i * C) + j]);
        }
        printf("\n");
      }
    }
    // --- --- 
    
    // Fazemos a multiplicação, gerando uma matriz H de forma (C, C):
    dim3 grid_dim_pm(ceil_div(n_heads, block_dim_x), ceil_div(C, block_dim_y), ceil_div(C, block_dim_z));
    dim3 block_dim_pm = block_dim_tl;
    primeira_multiplicacao<<<grid_dim_pm, block_dim_pm>>>(device_Q, device_K_transposto, device_H, C, C, D, sqrtD, n_heads);
    cudaDeviceSynchronize();

    // --- olhando produto interno ---
    // Fazendo esse produto na cpu
    double *host_H = (double *) malloc(n_heads * C * C * sizeof(double));
    produto_interno_escalado(host_Q, host_K_transposto, host_H, C, D, n_heads);

    double *device_copy_H_presoftmax = (double *) malloc(n_heads * C * C * sizeof(double));
    cudaMemcpy(device_copy_H_presoftmax, device_H, n_heads * C * C * sizeof(double), cudaMemcpyDeviceToHost);
    printf("Matrizes H presoftmax GPU:\n");
    for (int h = 0; h < n_heads; ++h) {
      printf("head: %d\n", h);
      for (int i = 0; i < C; ++i) {
        for (int j = 0; j < C; ++j){
          printf("%lf ", device_copy_H_presoftmax[(h * C * C) + (i * C) + j]);
        }
        printf("\n");
      }
    }
    
    printf("Matrizes H presoftmax CPU:\n");
    for (int h = 0; h < n_heads; ++h) {
      printf("head: %d\n", h);
      for (int i = 0; i < C; ++i) {
        for (int j = 0; j < C; ++j){
          printf("%lf ", host_H[(h * C * C) + (i * C) + j]);
        }
        printf("\n");
      }
    }
    // --- ---
    
    // Aplicamos o softmax na matriz resultante da última multiplicação:
    dim3 grid_dim_s(ceil_div(n_heads, block_dim_x), ceil_div(C, block_dim_y));
    dim3 block_dim_s(block_dim_x, block_dim_y);
    softmax<<<grid_dim_s, block_dim_s>>>(device_H, C, C, n_heads);
    cudaDeviceSynchronize();

    // --- Olhando softmax---
    softmax_cpu(host_H, C, n_heads);

    double *device_copy_H = (double *) malloc(n_heads * C * C * sizeof(double));
    cudaMemcpy(device_copy_H, device_H, n_heads * C * C * sizeof(double), cudaMemcpyDeviceToHost);

    for (int h = 0; h < n_heads; ++h) {
      for (int i = 0; i < C; ++i) {
        for (int j = 0; j < C; ++j) {
          if (fabs(host_H[(h * C * C) + (i * C) + j] - device_copy_H[(h * C * C) + (i * C) + j]) > 1e-9) {
            printf("Deu errado\n");
            printf("head: %d\n", h);
            printf("i: %d j: %d\n", i, j);
            printf("Valor na CPU: %lf\n", host_H[(h * C * C) + (i * C) + j]);
            printf("Valor na GPU: %lf\n", device_copy_H[(h * C * C) + (i * C) + j]);
            return;
          }
        }
      }
    }
    printf("Nada de errado no produto interno escalado!\n");
    printf("Matrizes H:\n");
    for (int h = 0; h < n_heads; ++h) {
      printf("head: %d\n", h);
      for (int i = 0; i < C; ++i) {
        for (int j = 0; j < C; ++j){
          printf("%lf ", host_H[(h * C * C) + (i * C) + j]);
        }
        printf("\n");
      }
    }
    // --- Fim do teste ---
  }
};


int main(){
  srand(998244353);

  double *E = (double *) malloc(C * d_model * sizeof(double));
  for (int i = 0; i < C * d_model; ++i) {
    E[i] = rand_double();
  }
  
  MultiHeadAttention Attention;
  Attention.pass_embedding(E);
}
