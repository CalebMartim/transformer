// Após calcular H = softmax(Q(K^T) / sqrt(D)),
// multiplicamos H por V, finalizando o 
// cálculo da função de atenção

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

// Transpõe uma matriz A de forma (n, m) e coloca o resultado 
// em B, que tem forma (m, n)
__global__ void transpor(double *A, double *B, int n, int m, int num_heads) {
  int h = threadIdx.x + blockDim.x + blockIdx.x;
  int i = threadIdx.y + blockDim.y * blockIdx.y;
  int j = threadIdx.z + blockDim.z * blockIdx.z;

  if (h < num_heads and i < n and j < m) {
    B[(h * m * n) + (j * n) + i] = A[(h * n * m) + (i * m) + j];
  }
}

// Multiplica uma matriz A de forma (n, k) com uma matriz B de forma (k, m) 
// e coloca o resultado na matriz C, que tem forma (n, m)
// Nesta função, aplicamos o conceito de masked self attention,
// para que nenum_headsum token obtenum_headsa informações sobre tokens em posições à frente
// e dividimos todo valor por D por questões de normalização de valores
__global__ void primeira_multiplicacao(double *A, double *B, double *C, int n, int m, int k, double sqrtD, int num_heads) {
  int h = threadIdx.x + blockDim.x * blockIdx.x;
  int j = threadIdx.y + blockDim.y * blockIdx.y;
  int i = threadIdx.z + blockDim.z * blockIdx.z;

  if (h < num_heads and j < m and i < n) {
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


// Pega uma matriz A de forma (n, m) e aplica a função softmax
// em cada uma de suas linum_headsas
__global__ void softmax(double *A, int n, int m, int num_heads){
  int h = threadIdx.x + blockDim.x * blockIdx.x;
  int i = threadIdx.y + blockDim.y * blockIdx.y;

  if (h < num_heads and i < n) {
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

// Multiplica uma matriz A de forma (n, k) com uma matriz B de forma (k, m) 
// e coloca o resultado na matriz C, que tem forma (n, m)
__global__ void segunda_multiplicacao(double *A, double *B, double *C, int n, int m, int k, int num_heads) {
  int h = threadIdx.x + blockDim.x * blockIdx.x;
  int j = threadIdx.y + blockDim.y * blockIdx.y;
  int i = threadIdx.z + blockDim.z * blockIdx.z;

  if (h < num_heads and j < m and i < n) {
    double soma = 0;
    for (int idx = 0; idx < k; ++idx) {
      soma += A[(h * n * k) + (i * k) + idx] * B[(h * k * m) + (idx * m) + j];
    }
    C[(h * n * m) + (i * m) + j] = soma;
  }
}

struct MultiHeadAttention{
  double *host_W_V, *host_W_Q, *host_W_K;
  double *device_W_V, *device_W_Q, *device_W_K, 
         *device_V, *device_Q, *device_K,
         *device_K_transposto;
  double *device_E, *device_H, *device_A;
  double *host_A;
  double *MultiHead;

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

    // Prepara a matriz transposta de K na GPU
    cudaMalloc(&device_K_transposto, num_heads * D * C * sizeof(double));

    // Matriz auxiliar para fazer a multiplicação entre Q e K^T
    cudaMalloc(&device_H, num_heads * C * C * sizeof(double));

    cudaMalloc(&device_A, num_heads * C * D * sizeof(double));
    
    // Aloca espaço para o resultado final do processo
    host_A = (double *) malloc(num_heads * C * D * sizeof(double));

    MultiHead = (double *) malloc(C * d_model * sizeof(double));
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

    double *host_V, *host_Q, *host_K;
    host_V = (double *) malloc(num_heads * C * D * sizeof(double));
    host_Q = (double *) malloc(num_heads * C * D * sizeof(double));
    host_K = (double *) malloc(num_heads * C * D * sizeof(double));

    transformacao_linear_cpu(host_W_V, E, host_V, C, d_model, D, num_heads);
    transformacao_linear_cpu(host_W_Q, E, host_Q, C, d_model, D, num_heads);
    transformacao_linear_cpu(host_W_K, E, host_K, C, d_model, D, num_heads);

    double *device_copy_V, *device_copy_Q, *device_copy_K;
    device_copy_V = (double *) malloc(num_heads * C * D * sizeof(double));
    device_copy_Q = (double *) malloc(num_heads * C * D * sizeof(double));
    device_copy_K = (double *) malloc(num_heads * C * D * sizeof(double));

    cudaMemcpy(device_copy_V, device_V, num_heads * C * D * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(device_copy_Q, device_Q, num_heads * C * D * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(device_copy_K, device_K, num_heads * C * D * sizeof(double), cudaMemcpyDeviceToHost);

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

    // Vamos transpor a matriz K para podermos fazer a multiplicação Q(K^T) 
    //dim3 grid_dim_t = grid_dim_tl;
    //dim3 block_dim_t = block_dim_tl;
    //transpor<<<grid_dim_t, block_dim_t>>>(device_K, device_K_transposto, C, D, num_heads);
    //cudaDeviceSynchronize();
    //
    //// Fazemos a multiplicação, gerando uma matriz H de forma (C, C):
    //dim3 grid_dim_pm(ceil_div(num_heads, block_dim_x), ceil_div(C, block_dim_y), ceil_div(C, block_dim_z));
    //dim3 block_dim_pm = block_dim_tl;
    //primeira_multiplicacao<<<grid_dim_pm, block_dim_pm>>>(device_Q, device_K_transposto, device_H, C, C, D, sqrtD, num_heads);
    //cudaDeviceSynchronize();

    //// Aplicamos o softmax na matriz resultante da última multiplicação:
    //dim3 grid_dim_s(ceil_div(num_heads, block_dim_x), ceil_div(C, block_dim_y));
    //dim3 block_dim_s(block_dim_x, block_dim_y);
    //softmax<<<grid_dim_s, block_dim_s>>>(device_H, C, C, num_heads);
    //cudaDeviceSynchronize();
    //
    //// Multiplicamos a matriz H, que tem forma (C, C), com a matriz V
    //// que tem forma (C, D) para finalizarmos o cálculo do attention
    //dim3 grid_dim_sm(ceil_div(num_heads, block_dim_x), ceil_div(D, block_dim_y), ceil_div(C, block_dim_z));
    //dim3 block_dim_sm = block_dim_tl;
    //segunda_multiplicacao<<<grid_dim_sm, block_dim_sm>>>(device_H, device_V, device_A, C, D, C, num_heads);
    //cudaDeviceSynchronize();
    //
    //printf("Não deu runtime error até agora!\n");
    //
    //cudaMemcpy(host_A, device_A, num_heads * C * D * sizeof(double), cudaMemcpyDeviceToHost);

    //for (int h = 0; h < num_heads; ++h) {
    //  for (int i = 0; i < C; ++i) {
    //    for (int j = 0; j < D; ++j) {
    //      MultiHead[(i * d_model) + (D * h) + j] = host_A[(h * C * D) + (i * D) + j];
    //      printf("%lf ", host_A[(h * C * D) + (i * D) + j]);
    //    }
    //    printf("\n");
    //  }
    //}

    //printf("resultado do Multi-Head attention:\n");
    //
    //for (int i = 0; i < C; ++i) {
    //  for (int j = 0; j < d_model; ++j) {
    //    printf("%lf ", MultiHead[(i * d_model) + j]);
    //  } 
    //  printf("\n");
    //}
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
