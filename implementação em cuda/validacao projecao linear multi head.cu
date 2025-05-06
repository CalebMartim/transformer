// Validação das projeções lineares feitas 
// com as matrizes de peso W_V, W_Q, W_K
// e a matriz de embedding E com 
// múltiplas cabeças

#include "stdio.h"
#include "cmath"

// constantes no modelo gpt-3, para referência
// const int d_model = 12288;
// const int num_heads = 96;
// const int C = 2048; // Tamanum_headso do context window
// const int vocabulary_size = 50257;
// const int D = d_model / num_heads;
// const int thread_size = 32;

// constantes meu modelo
const int d_model = 4;
const int num_heads = 2;
const int C = 2; // Tamanum_headso do context window
const int D = d_model / num_heads; // = 2
const int block_dim_x = 8, block_dim_y = 8, block_dim_z = 16;
const double sqrtD = sqrtl(D); 

// Teto da divisão de a por b
int ceil_div(int a, int b) {
  return (a + b - 1) / b;
}

// Pega um número aleatório entre -1 e 1
double rand_double() {
  double min = -1, max = 1;
  double range = (max - min);
  //return min + (double) rand() / (RAND_MAX / range);

  return rand() % 5;
}

// Aplica uma camada linear A de forma (k, m) a uma matriz B de forma (n, m)
// para obter uma matriz resultante C de forma (n, k)
__global__ void transformacao_linear(double *A, double *B, double *C, int n, int m, int k, int num_heads) {
  int h = threadIdx.x + blockDim.x * blockIdx.x;
  int i = threadIdx.y + blockDim.y * blockIdx.y;
  int j = threadIdx.z + blockDim.z * blockIdx.z;
  
  // i é o índice do vetor que vamos aplicar a camada agora
  // j é qual dimensão da camada estamos agora para aplicar a transformação
  if (h < num_heads and i < n and j < k) {
    double soma = 0;
    for (int idx = 0; idx < m; ++idx) {
      soma += A[(h * k * m) + (j * m) + idx] * B[(i * m) + idx];
    }
    C[(h * n * k) + (i * k) + j] = soma;
  }
}


// A: (k, m)
// B: (n, m)
// C: (n, k) 
void transformacao_linear_cpu(double *A, double *B, double *C, int n, int m, int k, int num_heads){
  for (int h = 0; h < num_heads; ++h) {
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < k; ++j) {
        double soma = 0;
        for (int idx = 0; idx < m; ++idx) {
          soma += A[(h * k * m) + (j * m) + idx] * B[(i * m) + idx];
        }
        C[(h * n * k) + (i * k) + j] = soma;
      }
    }
  }
}

struct MultiHeadAttention{
  double *host_W_V, *host_W_Q, *host_W_K;
  double *device_W_V, *device_W_Q, *device_W_K, 
         *device_V, *device_Q, *device_K;
  double *device_E;

  MultiHeadAttention(){
    // As seguintes são camadas lineares que transformam a
    // matriz de embedding da forma (C, d_model) para (C, D)
    host_W_V = (double *) malloc(num_heads * D * d_model * sizeof(double));
    host_W_Q = (double *) malloc(num_heads * D * d_model * sizeof(double));
    host_W_K = (double *) malloc(num_heads * D * d_model * sizeof(double));

    // Define inicialmente valores aleatórios para cada 
    // valor em W_V, W_Q, e W_K
    for (int i = 0; i < num_heads * D * d_model; ++i) {
      host_W_V[i] = rand_double();
      host_W_Q[i] = rand_double();
      host_W_K[i] = rand_double();
    }

    // Colocando essas matrizes na GPU
    cudaMalloc(&device_W_V, num_heads * D * d_model * sizeof(double));
    cudaMalloc(&device_W_Q, num_heads * D * d_model * sizeof(double));
    cudaMalloc(&device_W_K, num_heads * D * d_model * sizeof(double));
    cudaMemcpy(device_W_V, host_W_V, num_heads * D * d_model * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(device_W_Q, host_W_Q, num_heads * D * d_model * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(device_W_K, host_W_K, num_heads * D * d_model * sizeof(double), cudaMemcpyHostToDevice);

    // Alocando espaço para as matrizes 
    // V, Q, K. Estas serão as matrizes que
    // resultam das projeções da matriz de 
    // embedding nas camadas lineares
    cudaMalloc(&device_V, num_heads * C * D * sizeof(double));
    cudaMalloc(&device_Q, num_heads * C * D * sizeof(double));
    cudaMalloc(&device_K, num_heads * C * D * sizeof(double));

    // Aloca espaço para copiarmos a matriz de embedding para a GPU
    // (não precisa multiplicar por num_heads porque é único através das heads)
    cudaMalloc(&device_E, C * d_model * sizeof(double)); 
  }

  void pass_embedding(double *E){
    // E é o embedding do input. Ele tem forma (C, d_model),
    // onde C é o tamannum_headso da janela de contexto e d_model é 
    // a dimensão do modelo. Primeiro, temos que fazer uma 
    // transformação linear para transformar E de (C, d_model)
    // para (C, D), para termos exatamente C * d_model valores 
    // nos embeddings entre todas as heads. 

    // Copia E para a GPU
    cudaMemcpy(device_E, E, C * d_model * sizeof(double), cudaMemcpyHostToDevice);

    // Faz as transformações lineares paralelamente
    dim3 grid_dim_tl(ceil_div(num_heads, block_dim_x), ceil_div(C, block_dim_y), ceil_div(D, block_dim_z));
    dim3 block_dim_tl(block_dim_x, block_dim_y, block_dim_z);
    transformacao_linear<<<grid_dim_tl, block_dim_tl>>>(device_W_V, device_E, device_V, C, d_model, D, num_heads);
    transformacao_linear<<<grid_dim_tl, block_dim_tl>>>(device_W_Q, device_E, device_Q, C, d_model, D, num_heads);
    transformacao_linear<<<grid_dim_tl, block_dim_tl>>>(device_W_K, device_E, device_K, C, d_model, D, num_heads);
    cudaDeviceSynchronize();

    // --- Testando Transformações lineares: ---
    printf("Matriz de embedding:\n");
    for (int i = 0; i < C; ++i) {
      for (int j = 0; j < d_model; ++j) {
        printf("%lf ", E[(i * d_model) + j]);
      }
      printf("\n");
    }

    printf("Matrizes de pesos:\n");
    for (int h = 0; h < num_heads; ++h) {
      printf("Head: %d\n", h);
      printf("Matriz W_V:\n");
      for (int i = 0; i < D; ++i) {
        for (int j = 0; j < d_model; ++j) {
          printf("%lf ", host_W_V[(h * D * d_model) + (i * d_model) + j]);
        }
        printf("\n");
      }
      printf("Matriz W_Q:\n");
      for (int i = 0; i < D; ++i) {
        for (int j = 0; j < d_model; ++j) {
          printf("%lf ", host_W_Q[(h * D * d_model) + (i * d_model) + j]);
        }
        printf("\n");
      }
      printf("Matriz W_K:\n");
      for (int i = 0; i < D; ++i) {
        for (int j = 0; j < d_model; ++j) {
          printf("%lf ", host_W_K[(h * D * d_model) + (i * d_model) + j]);
        }
        printf("\n");
      }
    }

    // Resultados das transformações na CPU
    double *host_V, *host_Q, *host_K;
    host_V = (double *) malloc(num_heads * C * D * sizeof(double));
    host_Q = (double *) malloc(num_heads * C * D * sizeof(double));
    host_K = (double *) malloc(num_heads * C * D * sizeof(double));

    // Aplicação da transfornacao
    transformacao_linear_cpu(host_W_V, E, host_V, C, d_model, D, num_heads);
    transformacao_linear_cpu(host_W_Q, E, host_Q, C, d_model, D, num_heads);
    transformacao_linear_cpu(host_W_K, E, host_K, C, d_model, D, num_heads);

    // Copiando para o host o resultado calculado na GPU
    double *device_copy_V, *device_copy_Q, *device_copy_K;
    device_copy_V = (double *) malloc(num_heads * C * D * sizeof(double));
    device_copy_Q = (double *) malloc(num_heads * C * D * sizeof(double));
    device_copy_K = (double *) malloc(num_heads * C * D * sizeof(double));
    cudaMemcpy(device_copy_V, device_V, num_heads * C * D * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(device_copy_Q, device_Q, num_heads * C * D * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(device_copy_K, device_K, num_heads * C * D * sizeof(double), cudaMemcpyDeviceToHost);

    // Validação de fato:
    for (int h = 0; h < num_heads; ++h) {
      for (int i = 0; i < C; ++i) {
        for (int j = 0; j < D; ++j) {
          if (fabs(host_V[(h * C * D) + (i * D) + j] - device_copy_V[(h * C * D) + (i * D) + j]) > 1e-9) {
            printf("Erro em V\n");
            printf("Head: %d\n", h);
            printf("i: %d j: %d\n", i, j);
            printf("Valor na CPU: %lf\n", host_V[(h * C * D) + (i * D) + j]);
            printf("Valor na GPU: %lf\n", device_copy_V[(h * C * D) + (i * D) + j]);
            return;
          }
          if (fabs(host_Q[(h * C * D) + (i * D) + j] - device_copy_Q[(h * C * D) + (i * D) + j]) > 1e-9) {
            printf("Erro em Q\n");
            printf("Head: %d\n", h);
            printf("i: %d j: %d\n", i, j);
            printf("Valor na CPU: %lf\n", host_Q[(h * C * D) + (i * D) + j]);
            printf("Valor na GPU: %lf\n", device_copy_Q[(h * C * D) + (i * D) + j]);
            return;
          }
          if (fabs(host_K[(h * C * D) + (i * D) + j] - device_copy_K[(h * C * D) + (i * D) + j]) > 1e-9) {
            printf("Erro em K\n");
            printf("Head: %d\n", h);
            printf("i: %d j: %d\n", i, j);
            printf("Valor na CPU: %lf\n", host_K[(h * C * D) + (i * D) + j]);
            printf("Valor na GPU: %lf\n", device_copy_K[(h * C * D) + (i * D) + j]);
            return;
          }
        }
      }
    }

    printf("Nada de errado na projeção linear!\n");
    for (int h = 0; h < num_heads; ++h) {
      printf("Head: %d\n", h);
      printf("Matriz V:\n");
      for (int i = 0; i < C; ++i) {
        for (int j = 0; j < D; ++j) {
          printf("%lf ", host_V[(h * C * D) + (i * D) + j]);
        }
        printf("\n");
      }
      printf("Matriz Q:\n");
      for (int i = 0; i < C; ++i) {
        for (int j = 0; j < D; ++j) {
          printf("%lf ", host_Q[(h * C * D) + (i * D) + j]);
        }
        printf("\n");
      }
      printf("Matriz K:\n");
      for (int i = 0; i < C; ++i) {
        for (int j = 0; j < D; ++j) {
          printf("%lf ", host_K[(h * C * D) + (i * D) + j]);
        }
        printf("\n");
      }
    }
    // --- FIm do teste das projeções ---
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
