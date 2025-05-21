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

// W_X: (n_heads, D, d_model)
// E: (C, d_model)
// X: (n_heads, C, D)
void projecao_linear(double *W_X, double* E, double *X) {
  for (int h = 0; h < n_heads; ++h) {
    for (int i = 0; i < C; ++i) {
      for (int j = 0; j < D; ++j) {
        double soma = 0;
        for (int idx = 0; idx < d_model; ++idx) {
          soma += W_X[(h * C * d_model) + (j * d_model) + idx] * E[(i * d_model) + idx];
        }
        X[(h * C * D) + (i * D) + j] = soma;
      }
    }
  }
}

// k : (n_heads, C, D)
// K_transposto : (n_heads, D, C)
void transpor(double *K, double *K_transposto) {
  for (int h = 0; h < n_heads; ++h) {
    for (int i = 0; i < C; ++i) {
      for (int j = 0; j < D; ++j) {
        K_transposto[(h * D * C) + (j * C) + i] = K[(h * C * D) + (i * C) + j];
      }
    }
  }
}

// Q: (n_heads, C, D)
// K_tranposto: (n_heads, D, C)
// H: (n_heads, C, C)
void produto_interno(double *Q, double *K_transposto, double *H) {
  for (int h = 0; h < n_heads; ++h) {
    for (int j = 0; j < C; ++j) {
      for (int i = 0; i < C; ++i) {
        if (j <= i) {
          double soma = 0;
          for (int idx = 0; idx < D; ++idx) {
            soma += Q[(h * C * D) + (i * D) + idx] * K_transposto[(h * D * C) + (idx * C) + j];
          }
          H[(h * C * C) + (i * C) + j] = soma / sqrtD;
        } else {
          H[(h * C * C) + (i * C) + j] = -INFINITY;
        }
      }
    }
  }
}

// _H: (n_heads, C, C)
void softmax(double *H) {
  for (int h = 0; h < n_heads; ++h) {
    for (int i = 0; i < C; ++i) {
      double soma = 0;
      for (int idx = 0; idx < C; ++idx) {
        if (H[(h * C  * C) + (i * C) + idx] != -INFINITY) {
          soma += exp(H[(h * C  * C) + (i * C) + idx]);
        }
      }
      for (int idx = 0; idx < C; ++idx) {
        if (H[(h * C * C) + (i * C) + idx] == -INFINITY) {
          H[(h * C * C) + (i * C) + idx] = 0;
        } else {
          H[(h * C * C) + (i * C) + idx] = exp(H[(h * C * C) + (i * C) + idx]) / soma;
        }
      }
    }
  }
}

// H: (n_heads, C, C)
// V: (n_heads, C, D)
// A: (n_heads, C, D)
void matmul(double *H, double *V, double *A) {
  for (int h = 0; h < n_heads; ++h) {
    for (int j = 0; j < D; ++j) {
      for (int i = 0; i < C; ++i) {
        double soma = 0;
        for (int idx = 0; idx < C; ++idx) {
          soma += H[(h * C * C) + (i * C) + idx] * V[(h * C * D) + (idx * D) + j];
        }
        A[(h * C * D) + (i * D) + j] = soma;
      }
    }
  }
}

// A: (n_heads, C, D)
// A_concat: (C, d_model)
void concat(double *A, double *A_concat){
  for (int h = 0; h < n_heads; ++h) {
    for (int i = 0; i < C; ++i) {
      for (int j = 0; j < D; ++j) {
        A_concat[(i * n_heads * D) + (h * D) + j] = A[(h * C * D) + (i * C) + j];
      }
    }
  }
}

// concat: (C, d_model)
// W_O: (d_model, d_model)
// multihead: (C, d_model)
void projecao_final(double *concat, double *W_O, double *multihead) {
  for (int j = 0; j < d_model; ++j) {
    for (int i = 0; i < C; ++i) {
      double soma = 0;
      for (int idx = 0; idx < d_model; ++idx) {
        soma += concat[(i * d_model) + idx] * W_O[(idx * d_model) + j];
      } 
      multihead[(i * d_model) + j] = soma;
    }
  }
}


struct multihead_attention{
  double *W_V, *W_Q, *W_K, *W_O;

  multihead_attention(){
    W_V = (double *) malloc(n_heads * D * d_model * sizeof(double));
    W_Q = (double *) malloc(n_heads * D * d_model * sizeof(double));
    W_K = (double *) malloc(n_heads * D * d_model * sizeof(double));
    W_O = (double *) malloc(d_model * d_model * sizeof(double));
    
    for (int i = 0; i < n_heads * D * d_model; ++i) {
      W_V[i] = rand_double();
      W_Q[i] = rand_double();
      W_K[i] = rand_double();
      W_O[i] = rand_double();
    }
  }

  double* pass(double *E) {

    double *V, *Q, *K;
    V = (double *) malloc(n_heads * C * D * sizeof(double));
    Q = (double *) malloc(n_heads * C * D * sizeof(double));
    K = (double *) malloc(n_heads * C * D * sizeof(double));

    projecao_linear(W_V, E, V);
    projecao_linear(W_Q, E, Q);
    projecao_linear(W_K, E, K);

    printf("V:\n");
    for (int i = 0; i < n_heads; ++i) {
      printf("Head %d:\n", i);
      for (int j = 0; j < C; ++j) {
        for (int k = 0; k < D; ++k) {
          printf("%lf ", V[(i * C * D) + (j * D) + k]);
        } 
        printf("\n");
      }
    }
    printf("\n");

    printf("Q:\n");
    for (int i = 0; i < n_heads; ++i) {
      printf("Head %d:\n", i);
      for (int j = 0; j < C; ++j) {
        for (int k = 0; k < D; ++k) {
          printf("%lf ", Q[(i * C * D) + (j * D) + k]);
        } 
        printf("\n");
      }
    }
    printf("\n");

    printf("K:\n");
    for (int i = 0; i < n_heads; ++i) {
      printf("Head %d:\n", i);
      for (int j = 0; j < C; ++j) {
        for (int k = 0; k < D; ++k) {
          printf("%lf ", K[(i * C * D) + (j * D) + k]);
        } 
        printf("\n");
      }
    }
    printf("\n");

    // Transpõe a matriz K
    double *K_transposto = (double*) malloc(n_heads * D * C * sizeof(double));
    transpor(K, K_transposto);
    
    printf("K transposto:\n");
    for (int h = 0; h < n_heads; ++h) {
      printf("Head %d:\n", h);
      for (int i = 0; i < D; ++i) {
        for (int j = 0; j < C; ++j) {
          printf("%lf ", K_transposto[(h * D * C) + (i * C) + j]);
        }
        printf("\n");
      }
    }
    printf("\n");
    
    // Multiplicação Q por K^T
    double *H = (double *) malloc(n_heads * C * C * sizeof(double));
    produto_interno(Q, K_transposto, H);

    printf("H:\n");
    for (int h = 0; h < n_heads; ++h) {
      for (int i = 0; i < C; ++i) {
        for (int j = 0; j < C; ++j) {
          printf("%lf ", H[(h * C * C) + (i * C) + j]);
        }
        printf("\n");
      }
    }
    printf("\n");

    // Aplica softmax em H
    softmax(H);
    
    printf("Softmax:\n");
    for (int h = 0; h < n_heads; ++h) {
      for (int i = 0; i < C; ++i) {
        for (int j = 0; j < C; ++j) {
          printf("%lf ", H[(h * C * C) + (i * C) + j]);
        }
        printf("\n");
      }
    }
    printf("\n");

    // Obtém Attention em cada head 
    double *A = (double *) malloc(n_heads * C * D * sizeof(double));
    matmul(H, V, A);

    printf("Device A:\n");
    for (int h = 0; h < n_heads; ++h) {
      for (int i = 0; i < C; ++i) {
        for (int j = 0; j < D; ++j) {
          printf("%lf ", A[(h * C * D) + (i * D) + j]);
        }
        printf("\n");
      }
    }
    printf("\n");

    // Concatena tudo em concat
    double *A_concat = (double *) malloc(C * d_model * sizeof(double));
    concat(A, A_concat);

      
    printf("Concat:\n");
    for (int i = 0; i < C; ++i) {
      for (int j = 0; j < d_model; ++j) {
        printf("%lf ", A_concat[(i * d_model) + j]);
      }
      printf("\n");
    }
    printf("\n");

    // Calcula o resultado final de multihead
    double *multihead = (double *) malloc(C * d_model * sizeof(double)); 
    projecao_final(A_concat, W_O, multihead);

    return multihead;
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

