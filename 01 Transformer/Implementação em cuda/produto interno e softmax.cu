// Após aplicar as camadas lineares na matrix de embedding,
// gerando Q, K, V, calculamos softmax( Q(K^T) / sqrt(D) )

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

void transpor_cpu(double *A, double *B, int n, int m) {
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < m; ++j) {
      B[j * n + i] = A[i * m + j];
    }
  }
}

// Multiplica uma matriz A de forma (n, k) com uma matriz B de forma (k, m) 
// e coloca o resultado na matriz C, que tem forma (n, m)
__global__ void multiplica_matrizes(double *A, double *B, double *C, int n, int m, int k, double sqrtD) {
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

void multiplicar_cpu(double *A, double *B, double *C, int n, int m, int k) {
  for (int j = 0; j < m; ++j) {
    for (int i = 0; i < n; ++i) {
      if (j <= i) {
        double soma = 0;
        for (int idx = 0; idx < k; ++idx) {
          soma += A[i * k + idx] * B[idx * m + j];
        }
        C[i * m + j] = soma / sqrtD; 
      } else {
        C[i * m + j] = -INFINITY; 
      }
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

void softmax_cpu(double *A, int n, int m) {
  for (int i = 0; i < n; ++i) {
    double soma = 0;
    for (int idx = 0; idx < m; ++idx) {
      if (A[i * m + idx] != -INFINITY) {
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
    dim3 grid_dim_m(ceil_div(C, thread_size), ceil_div(C, thread_size));
    dim3 block_dim_m = block_dim_tl;
    multiplica_matrizes<<<grid_dim_m, block_dim_m>>>(device_Q, device_K_transposto, device_H, C, C, D, sqrtD);
    cudaDeviceSynchronize();

    // Aplicamos o softmax na matriz resultante da última multiplicação:
    int grid_dim_s = ceil_div(C, thread_size);
    int block_dim_s = thread_size;
    softmax<<<grid_dim_s, block_dim_s>>>(device_H, C, C);
    cudaDeviceSynchronize();
    
    // --- Testando validade da multiplicação e softmax --- 
    // Aloca espaço e define variáveis
    double *host_Q = (double *) malloc(C * D * sizeof(double)),
           *host_K = (double *) malloc(C * D * sizeof(double)),
           *host_K_transposto = (double *) malloc(D * C * sizeof(double)),
           *host_H = (double *) malloc(C * C * sizeof(double));

    // Copia as matrizes Q e K que já verificamos estarem corretos
    cudaMemcpy(host_Q, device_Q, C * D * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_K, device_K, C * D * sizeof(double), cudaMemcpyDeviceToHost);

    transpor_cpu(host_K, host_K_transposto, C, D);
    
    multiplicar_cpu(host_Q, host_K_transposto, host_H, C, C, D);
    
    // Olhando o resultado da multiplicação antes de fazer o softmax
    printf("Resultado da multiplicação (pré-softmax):\n");
    for (int i = 0; i < C; ++i) {
      for (int j = 0; j < C; ++j) {
        printf("%lf ", host_H[i * C + j]);
      }
      printf("\n");
    }

    softmax_cpu(host_H, C, C);

    // Copia o valor de H calculado na GPU
    double *device_copy_H = (double *) malloc(C * C * sizeof(double));
    cudaMemcpy(device_copy_H, device_H, C * C * sizeof(double), cudaMemcpyDeviceToHost);
    // Faz a verificação dos valores
    for (int i = 0; i < C; ++i) {
      for (int j = 0; j < C; ++j) {
        if (fabs(host_H[i * C + j] - device_copy_H[i * C + j]) > 1e-9) {
          printf("Deu errado\n");
          printf("i: %d j: %d\n", i, j);
          printf("Valor na CPU: %lf\n", host_H[i * C + j]);
          printf("Valor na GPU: %lf\n", device_copy_H[i * C + j]);
          return;
        }
      }
    }
    printf("Deu certo!\n");
    printf("Resultado da matriz H (pós-softmax):\n");
    for (int i = 0; i < C; ++i) {
      for (int j = 0; j < C; ++j) {
        printf("%lf ", host_H[i * C + j]);
      }
      printf("\n");
    }
    // --- Fim do teste da multiplicação e softmax--
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
