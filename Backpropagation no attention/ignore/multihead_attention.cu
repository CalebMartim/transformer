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

const int d_model = 4; 
const int n_heads = 2;
const int C = 2; // Janela de contexto
const int D = d_model / n_heads; // d_k 
const int block_size_x = 8, block_size_y = 8, block_size_z = 16;

#include "stdio.h"
#include "cmath"

#include <cmath>
#include <cassert>
#include <vector>
#include <unordered_set>
#include <algorithm>

using namespace std;
using ld = double;

// [Funções auxiliares:]
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
// [Fim de funções auxiliares]


// struct valor; 
// vector<valor> valores;
// vector<ld> grad;

struct valor{
  ld data;
  valor* left_child, *right_child;  
  int op;
  ld expoente;
  ld grad = 0;

  __device__ __host__ 
  valor(ld _data = 0, valor *_left_child = nullptr, valor * _right_child = nullptr, int _op = 0, ld _expoente = 0) {
    data = _data;
    left_child = _left_child;
    right_child = _right_child;
    op = _op;
    expoente = _expoente;
    grad = 0;
  }

  __device__ __host__ valor operator-(valor &b) {
   valor x(-1);
   valor neg = b * x;
   return (*this) + neg;
  }

  __device__ __host__ valor _pow(ld k) {
    return valor(pow(data, k), this, nullptr, 3, k);
  }

  __device__ __host__ valor operator/(valor b) {
   valor x = b._pow(-1);
   return (*this) * x;
  }

  __device__ __host__ valor _exp() {
    return valor(exp(data), this, nullptr, 4);
  }

  void prop() {
    if (op == 1) {
      left_child->grad += grad;
      right_child->grad += grad;
    } else if (op == 2) {
      left_child->grad += right_child->data * grad;
      right_child->grad += left_child->data * grad;
    } else if (op == 3) {
      left_child->grad += expoente * powl(left_child->data, expoente - 1) * grad;
    } else if (op == 4) {
      left_child->grad += data * grad;
    } 
  }

  void backward_pass(){
    grad = 1;
    
    // Vamos fazer aqui a ordenção topológica:
    vector<valor*> top_sort;
    unordered_set<valor*> vis;
    auto dfs = [&](valor *v, auto &&self) -> void {
      printf("%p\n", v);
      if (vis.count(v)) return;
      vis.insert(v);
      if(v->left_child) {
        self(v->left_child, self);
        if (v->right_child) {
          self(v->right_child, self);
        }
      }
      top_sort.push_back(v);
    };
    
    dfs(this, dfs);
    reverse(top_sort.begin(), top_sort.end());

    for (valor *v : top_sort) {
      v->prop();
    }
  }
};

struct ValorArena {
  valor* pool;
  int capacity;
  int *index; 
  
  __device__ 
  valor *alloc(double x, valor *left, valor *right, int op = 0, double k = 0){
    int i = atomicAdd(index, 1);
    if (i < capacity) {
      return new (&pool[i]) valor(x, left, right, op, k);
    }
    return nullptr;
  } 
};

__device__ 
valor* operator+(valor &a, valor &b, ValorArena & arena) {
  valor* out;
  cudaMallocManaged(&out, sizeof(valor));
  new (out) valor(a.data + b.data, &a, &b, 1);
  return out;
}

__device__ __host__ 
valor* operator*(valor &a, valor &b) {
  valor* out;
  cudaMallocManaged(&out, sizeof(valor));
  new (out) valor(a.data * b.data, &a, &b, 2);
  return out;
}

__device__ __host__ 
valor* operator-(valor &a) {
  valor *aux;
  cudaMallocManaged(&aux, sizeof(valor));
  new (aux) valor(-1);
  return a * (*aux);
}


const valor sqrtD = sqrtl(D);

// device_W_x: (n_heads, D, d_model)
// device_E: (C, d_model) 
// device_x: (n_heads, C, D)
__global__ void projecao_linear(valor *device_W_X, valor *device_E, valor *device_X, int C, int D, int d_model, int n_heads){
  int h = threadIdx.x + (blockDim.x * blockIdx.x); // head atual
  int i = threadIdx.y + (blockDim.y * blockIdx.y); // linha do device_E
  int j = threadIdx.z + (blockDim.z * blockIdx.z); // linha do device_W_X
  
  if (h < n_heads and i < C and j < D) {
    valor soma = 0;
    for (int idx = 0; idx < d_model; ++idx) {
      valor n = device_W_X[(h * D * d_model) + (j * d_model) + idx] * device_E[(i * d_model) + idx];
      valor aux = soma;
      soma = *(aux + n);
    }
    device_X[(h * C * D) + (i * D) + j] = soma;
  }
}

// k : (n_heads, C, D)
// K_t : (n_heads, D, C)
__global__ void transpor(valor* device_K, valor *device_K_transposto, int C, int D, int n_heads){
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
__global__ void produto_interno(valor *Q, valor *K_transposto, valor *device_H, int C, int D, int n_heads, valor sqrtD) {
  int h = threadIdx.x + (blockDim.x * blockIdx.x);
  int j = threadIdx.y + (blockDim.y * blockIdx.y);
  int i = threadIdx.z + (blockDim.z * blockIdx.z);

  if (h < n_heads and j < C and i < C) {
    if (j <= i) {
      valor soma = 0;
      for (int idx = 0; idx < D; ++idx) {
        valor m = Q[(h * C * D) + (i * D) + idx] * K_transposto[(h * D * C) + (idx * C) + j];
        soma = soma + m;
      }
      device_H[(h * C * C) + (i * C) + j] = soma / sqrtD;
    } else {
      device_H[(h * C * C) + (i * C) + j] = -INFINITY;
    }
  }
}

// device_HL: (n_heads, C, C)
__global__ void softmax(valor *device_H, int C, int n_heads){
  int h = threadIdx.x + (blockDim.x * blockIdx.x);
  int i = threadIdx.y + (blockDim.y * blockIdx.y);
  
  if (h < n_heads and i < C) {
    valor soma = 0;
    for (int idx = 0; idx < C; ++idx) {
      if (device_H[(h * C * C) + (i * C) + idx].data != -INFINITY) {
        valor x = device_H[(h * C * C) + (i * C) + idx]._exp();
        soma = soma + x;
      }
    }
    for (int idx = 0; idx < C; ++idx) {
      if (device_H[(h * C * C) + (i * C) + idx].data != -INFINITY) {
        valor e = device_H[(h * C * C) + (i * C) + idx]._exp();
        device_H[(h * C * C) + (i * C) + idx] =  e / soma;
      } else {
        device_H[(h * C * C) + (i * C) + idx] = 0;
      }
    }
  }
}

// device_H: (n_heads, C, C)
// device_V: (n_heads, C, D)
// device_A: (n_heads, C, D)
__global__ void matmul(valor *device_H, valor *device_V, valor *device_A, int C, int D, int n_heads){
  int h = threadIdx.x + (blockDim.x * blockIdx.x); // head atual
  int j = threadIdx.y + (blockDim.y * blockIdx.y); // coluna de V
  int i = threadIdx.z + (blockDim.z * blockIdx.z); // linha de H
                                                 
  if (h < n_heads and j < D and i < C) {
    valor soma = 0;
    for (int idx = 0; idx < C; ++idx) {
      valor n = device_H[(h * C * C) + (i * C) + idx] * device_V[(h * C * D) + (idx * D) + j];
      soma = soma + n;
    }
    device_A[(h * C * D) + (i * D) + j] = soma;
  }                                                 
}

// device_A: (n_heads, C, D)
// deice_concat: (C, d_model)
__global__ void concat(valor *device_A, valor *device_concat, int C, int D, int n_heads){
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
__global__ void projecao_final(valor *device_concat, valor *device_W_O, valor* multihead, int C, int d_model){
  int j = threadIdx.x + (blockIdx.x * blockDim.x);
  int i = threadIdx.y + (blockIdx.y * blockDim.y);
  
  if (j < d_model and i < C) {
    valor soma = 0;
    for (int idx = 0; idx < d_model; ++idx) {
      valor m = device_concat[(i * d_model) + idx] * device_W_O[(idx * d_model) + j];
      soma = soma + m;
    }
    multihead[(i * d_model) + j] = soma;
  }
}


struct multihead_attention{
  valor *host_W_V, *host_W_Q, *host_W_K, *host_W_O;
  valor *device_W_V, *device_W_Q, *device_W_K, *device_W_O;

  valor *host_multihead;

  multihead_attention(){
    host_W_V = (valor *) malloc(n_heads * D * d_model * sizeof(valor));
    host_W_Q = (valor *) malloc(n_heads * D * d_model * sizeof(valor));
    host_W_K = (valor *) malloc(n_heads * D * d_model * sizeof(valor));
    host_W_O = (valor *) malloc(d_model * d_model * sizeof(valor));
    
    for (int i = 0; i < n_heads * D * d_model; ++i) {
      host_W_V[i] = rand_double();
      host_W_Q[i] = rand_double();
      host_W_K[i] = rand_double();
      host_W_O[i] = rand_double();
    }

    cudaMalloc(&device_W_V, n_heads * D * d_model * sizeof(valor));
    cudaMalloc(&device_W_Q, n_heads * D * d_model * sizeof(valor));
    cudaMalloc(&device_W_K, n_heads * D * d_model * sizeof(valor));
    cudaMalloc(&device_W_O, d_model * d_model * sizeof(valor));

    cudaMemcpy(device_W_V, host_W_V, n_heads * D * d_model * sizeof(valor), cudaMemcpyHostToDevice);
    cudaMemcpy(device_W_Q, host_W_Q, n_heads * D * d_model * sizeof(valor), cudaMemcpyHostToDevice);
    cudaMemcpy(device_W_K, host_W_K, n_heads * D * d_model * sizeof(valor), cudaMemcpyHostToDevice);
    cudaMemcpy(device_W_O, host_W_O, d_model * d_model * sizeof(valor), cudaMemcpyHostToDevice);

    host_multihead = (valor *) malloc(C * d_model * sizeof(valor));
  }

  valor* pass(valor *E) {

    // Passando a matriz de embedding para a gpu
    valor *device_E;
    cudaMalloc(&device_E, C * d_model * sizeof(valor));
    cudaMemcpy(device_E, E, C * d_model * sizeof(valor), cudaMemcpyHostToDevice);

    valor *device_V, *device_Q, *device_K;
    cudaMalloc(&device_V, n_heads * C * D * sizeof(valor));
    cudaMalloc(&device_Q, n_heads * C * D * sizeof(valor));
    cudaMalloc(&device_K, n_heads * C * D * sizeof(valor));

    // Cálculo das projeções lineares
    dim3 grid_dim_pl(ceil_div(n_heads, block_size_x), ceil_div(C, block_size_y), ceil_div(D, block_size_z));
    dim3 block_dim_pl(block_size_x, block_size_y, block_size_z);
    projecao_linear<<<grid_dim_pl, block_dim_pl>>>(device_W_V, device_E, device_V, C, D, d_model, n_heads);
    projecao_linear<<<grid_dim_pl, block_dim_pl>>>(device_W_Q, device_E, device_Q, C, D, d_model, n_heads);
    projecao_linear<<<grid_dim_pl, block_dim_pl>>>(device_W_K, device_E, device_K, C, D, d_model, n_heads);
    cudaDeviceSynchronize();

    valor *host_V = (valor *) malloc(n_heads * C * D * sizeof(valor));
    cudaMemcpy(host_V, device_V, n_heads * C * D * sizeof(valor), cudaMemcpyDeviceToHost);

    printf("V:\n");
    for (int i = 0; i < n_heads; ++i) {
      printf("Head %d:\n", i);
      for (int j = 0; j < C; ++j) {
        for (int k = 0; k < D; ++k) {
          printf("%lf ", host_V[(i * C * D) + (j * D) + k].data);
        } 
        printf("\n");
      }
    }
    printf("\n");

    valor *host_Q = (valor *) malloc(n_heads * C * D * sizeof(valor));
    cudaMemcpy(host_Q, device_Q, n_heads * C * D * sizeof(valor), cudaMemcpyDeviceToHost);
    printf("Q:\n");
    for (int i = 0; i < n_heads; ++i) {
      printf("Head %d:\n", i);
      for (int j = 0; j < C; ++j) {
        for (int k = 0; k < D; ++k) {
          printf("%lf ", host_Q[(i * C * D) + (j * D) + k].data);
        } 
        printf("\n");
      }
    }
    printf("\n");

    valor *host_K = (valor *) malloc(n_heads * C * D * sizeof(valor));
    cudaMemcpy(host_K, device_K, n_heads * C * D * sizeof(valor), cudaMemcpyDeviceToHost);
    printf("K:\n");
    for (int i = 0; i < n_heads; ++i) {
      printf("Head %d:\n", i);
      for (int j = 0; j < C; ++j) {
        for (int k = 0; k < D; ++k) {
          printf("%lf ", host_K[(i * C * D) + (j * D) + k].data);
        } 
        printf("\n");
      }
    }
    printf("\n");

    // Transpõe a matriz K
    valor *device_K_transposto;
    cudaMalloc(&device_K_transposto, n_heads * D * C * sizeof(valor));
    dim3 grid_dim_t = grid_dim_pl;
    dim3 block_dim_t = block_dim_pl;
    transpor<<<grid_dim_t, block_dim_t>>>(device_K, device_K_transposto, C, D, n_heads);
    cudaDeviceSynchronize();
    
    valor *host_K_transposto = (valor *) malloc(n_heads * D * C * sizeof(valor));
    cudaMemcpy(host_K_transposto, device_K_transposto, n_heads * D * C * sizeof(valor), cudaMemcpyDeviceToHost);
    printf("K transposto:\n");
    for (int h = 0; h < n_heads; ++h) {
      printf("Head %d:\n", h);
      for (int i = 0; i < D; ++i) {
        for (int j = 0; j < C; ++j) {
          printf("%lf ", host_K_transposto[(h * D * C) + (i * C) + j].data);
        }
        printf("\n");
      }
    }
    printf("\n");
    
    // Multiplicação Q por K^T
    valor *device_H;
    cudaMalloc(&device_H, n_heads * C * C * sizeof(valor));
    dim3 grid_dim_pi(ceil_div(n_heads, block_size_x), ceil_div(C, block_size_y), ceil_div(C, block_size_z));
    dim3 block_dim_pi = block_dim_pl;
    produto_interno<<<grid_dim_pi, block_dim_pi>>>(device_Q, device_K_transposto, device_H, C, D, n_heads, sqrtD);
    cudaDeviceSynchronize();

    valor *host_H = (valor *) malloc(n_heads * C * C * sizeof(valor));
    cudaMemcpy(host_H, device_H, n_heads * C * C * sizeof(valor), cudaMemcpyDeviceToHost);
    printf("Device H\n");
    for (int h = 0; h < n_heads; ++h) {
      for (int i = 0; i < C; ++i) {
        for (int j = 0; j < C; ++j) {
          printf("%lf ", host_H[(h * C * C) + (i * C) + j].data);
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
    
    cudaMemcpy(host_H, device_H, n_heads * C * C * sizeof(valor), cudaMemcpyDeviceToHost);
    printf("Softmax:\n");
    for (int h = 0; h < n_heads; ++h) {
      for (int i = 0; i < C; ++i) {
        for (int j = 0; j < C; ++j) {
          printf("%lf ", host_H[(h * C * C) + (i * C) + j].data);
        }
        printf("\n");
      }
    }
    printf("\n");

    // Obtém Attention em cada head 
    valor *device_A;
    cudaMalloc(&device_A, n_heads * C * D * sizeof(valor));
    dim3 grid_dim_mm(ceil_div(n_heads, block_size_x), ceil_div(D, block_size_y), ceil_div(C, block_size_z));
    dim3 block_dim_mm = block_dim_pl;
    matmul<<<grid_dim_mm, block_dim_mm>>>(device_H, device_V, device_A, C, D, n_heads);
    cudaDeviceSynchronize();

    valor *host_A = (valor *) malloc(n_heads * C * D * sizeof(valor));
    cudaMemcpy(host_A, device_A, n_heads * C * D * sizeof(valor), cudaMemcpyDeviceToHost);
    printf("Device A:\n");
    for (int h = 0; h < n_heads; ++h) {
      for (int i = 0; i < C; ++i) {
        for (int j = 0; j < D; ++j) {
          printf("%lf ", host_A[(h * C * D) + (i * D) + j].data);
        }
        printf("\n");
      }
    }
    printf("\n");

    // Concatena tudo em concat
    valor *device_concat;
    cudaMalloc(&device_concat, C * d_model * sizeof(valor));
    dim3 grid_dim_c = grid_dim_pl;
    dim3 block_dim_c = block_dim_pl;
    concat<<<grid_dim_c, block_dim_c>>> (device_A, device_concat, C, D, n_heads);
    cudaDeviceSynchronize();

    valor *host_concat = (valor *) malloc(C * d_model * sizeof(valor));
    cudaMemcpy(host_concat, device_concat, C * d_model * sizeof(valor), cudaMemcpyDeviceToHost);
      
    printf("Concat:\n");
    for (int i = 0; i < C; ++i) {
      for (int j = 0; j < d_model; ++j) {
        printf("%lf ", host_concat[(i * d_model) + j].data);
      }
      printf("\n");
    }
    printf("\n");

    // Calcula o resultado final de multihead
    valor *multihead; 
    cudaMalloc(&multihead, C * d_model * sizeof(valor));
    dim3 grid_dim_pf(ceil_div(d_model, block_size_x), ceil_div(C, block_size_y));
    dim3 block_dim_pf = block_dim_s;
    projecao_final<<<grid_dim_pf, block_dim_pf>>>(device_concat, device_W_O, multihead, C, d_model);
    cudaDeviceSynchronize();

    cudaMemcpy(host_multihead, multihead, C * d_model * sizeof(valor), cudaMemcpyDeviceToHost);
    
    return host_multihead;
  }
};

int main(){
  srand(998244353);

  valor *E = (valor *) malloc(C * d_model * sizeof(valor));
  for (int i = 0; i < C * d_model; ++i) {
    E[i] = rand_double();
  }

  multihead_attention mha;
  valor *res =  mha.pass(E);

  printf("Resultado do attention:\n");

  valor custo = 0;
  for (int i = 0; i < C; ++i) {
    for (int j = 0; j < d_model; ++j) {
      printf("%lf ", res[(i * d_model) + j].data);
      custo = valor(custo.data) + res[(i * d_model) + j];
    }
    printf("\n");
  }
  printf("%lf\n", custo.data);
  custo.backward_pass();
  printf("Não quebrou\n");
}

