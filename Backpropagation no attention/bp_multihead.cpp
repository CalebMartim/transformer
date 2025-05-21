#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cassert>

using namespace std;

struct valor; 
vector<valor> valores;
vector<double> grad;

struct valor{
  double data;
  int left_child, right_child;  
  int op;
  int id; 
  double expoente;

  valor(double _data = 0, int _left_child = -1, int _right_child = -1, int _op = 0, double _expoente = 0) {
    this->data = _data;
    this->left_child = _left_child;
    this->right_child = _right_child;
    this->op = _op;
    this->id = (int) size(valores);
    this->expoente = _expoente;
    // Faz a cópia deste valor na nossa "memória"
    valores.push_back(*this); 
    // Gradiente desse valor, inicialmente 0
    grad.push_back(0);
  }

  // Mostra em algum outputstream
  friend ostream& operator<<(ostream& os, valor & v) {
    os << v.data;
    return os;
  }

  valor operator+(valor b) {
    return valor(data + b.data, id, b.id, 1);
  }
  
  valor operator+(double b) {
    valor x(b);
    return (*this) + x;
  }

  friend valor operator+(double x, valor a) {
    return a + x;
  }

  valor operator*(valor b) {
    return valor(data * b.data, id, b.id, 2);
  }

  valor operator*(double b) {
    valor x(b);
    return (*this) * x;
  }

  friend valor operator*(double b, valor a) {
    return a * b;
  }
  
  valor operator-() {
    return (*this) * -1;
  }

  valor operator-(valor b) {
   return (*this) + -b;
  }

  friend valor operator-(double b, valor a) {
    return b + -a;
  }

  valor vpow(double k) {
    return valor(pow(data, k), id, -1, 3, k);
  }

  valor operator/(valor b) {
   return (*this) * b.vpow(-1);
  }

  valor operator/(double b) {
   return (*this) * (1 / b);
  }

  friend valor operator/(double b, valor a) {
   return b * a.vpow(-1);
  }

  valor vexp() {
    return valor(exp(data), id, -1, 4);
  }

  valor tanh() {
    double z = (exp(2 * data) - 1) / (exp(2 * data) + 1);
    return valor(z, id, -1, 5); 
  }

  // Propagação dos gradientes
  void prop() {
    if (op == 1) {
      grad[left_child] += grad[id];
      grad[right_child] += grad[id];
    } else if (op == 2) {
      grad[left_child] += valores[right_child].data * grad[id];
      grad[right_child] += valores[left_child].data * grad[id];
    } else if (op == 3) {
      grad[left_child] += expoente * pow(valores[left_child].data, expoente - 1) * grad[id];
    } else if (op == 4) {
      grad[left_child] += data * grad[id];
    } else if (op == 5) {
      grad[left_child] += (1 - data * data) * grad[id];
    }
  }

  // Função que calcula os gradientes a partir deste valor
  void backward_pass(){
    grad[id] = 1;
    
    // Vamos fazer aqui a ordenção topológica:
    vector<int> top_sort;
    vector<bool> vis(size(valores));
    auto dfs = [&](int v, auto &&self) -> void {
      if (v == -1 or vis[v]) return;
      vis[v] = true;
      if(valores[v].left_child) {
        self(valores[v].left_child, self);
        if (valores[v].right_child) {
          self(valores[v].right_child, self);
        }
      }
      top_sort.push_back(v);
    };
    
    dfs(id, dfs);
    reverse(top_sort.begin(), top_sort.end());

    for (int v : top_sort) {
      valores[v].prop();
    }
  }
};


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
void projecao_linear(valor *W_X, valor *E, valor *X) {
  for (int h = 0; h < n_heads; ++h) {
    for (int i = 0; i < C; ++i) {
      for (int j = 0; j < D; ++j) {
        valor soma = 0;
        for (int idx = 0; idx < d_model; ++idx) {
          soma = soma + W_X[(h * C * d_model) + (j * d_model) + idx] * E[(i * d_model) + idx];
        }
        X[(h * C * D) + (i * D) + j] = soma;
      }
    }
  }
}

// k : (n_heads, C, D)
// K_transposto : (n_heads, D, C)
void transpor(valor *K, valor *K_transposto) {
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
void produto_interno(valor *Q, valor *K_transposto, valor *H) {
  for (int h = 0; h < n_heads; ++h) {
    for (int j = 0; j < C; ++j) {
      for (int i = 0; i < C; ++i) {
        if (j <= i) {
          valor soma = 0;
          for (int idx = 0; idx < D; ++idx) {
            soma = soma + Q[(h * C * D) + (i * D) + idx] * K_transposto[(h * D * C) + (idx * C) + j];
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
void softmax(valor *H) {
  for (int h = 0; h < n_heads; ++h) {
    for (int i = 0; i < C; ++i) {
      valor soma = 0;
      for (int idx = 0; idx < C; ++idx) {
        if (H[(h * C  * C) + (i * C) + idx].data != -INFINITY) {
          soma = soma + H[(h * C  * C) + (i * C) + idx].vexp();
        }
      }
      for (int idx = 0; idx < C; ++idx) {
        if (H[(h * C * C) + (i * C) + idx].data == -INFINITY) {
          H[(h * C * C) + (i * C) + idx] = 0;
        } else {
          H[(h * C * C) + (i * C) + idx] = H[(h * C * C) + (i * C) + idx].vexp() / soma;
        }
      }
    }
  }
}

// H: (n_heads, C, C)
// V: (n_heads, C, D)
// A: (n_heads, C, D)
void matmul(valor *H, valor *V, valor *A) {
  for (int h = 0; h < n_heads; ++h) {
    for (int j = 0; j < D; ++j) {
      for (int i = 0; i < C; ++i) {
        valor soma = 0;
        for (int idx = 0; idx < C; ++idx) {
          soma = soma + H[(h * C * C) + (i * C) + idx] * V[(h * C * D) + (idx * D) + j];
        }
        A[(h * C * D) + (i * D) + j] = soma;
      }
    }
  }
}

// A: (n_heads, C, D)
// A_concat: (C, d_model)
void concat(valor *A, valor *A_concat){
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
void projecao_final(valor *concat, valor *W_O, valor *multihead) {
  for (int j = 0; j < d_model; ++j) {
    for (int i = 0; i < C; ++i) {
      valor soma = 0;
      for (int idx = 0; idx < d_model; ++idx) {
        soma = soma + concat[(i * d_model) + idx] * W_O[(idx * d_model) + j];
      } 
      multihead[(i * d_model) + j] = soma;
    }
  }
}


struct multihead_attention{
  valor *W_V, *W_Q, *W_K, *W_O;

  multihead_attention(){
    W_V = (valor *) malloc(n_heads * D * d_model * sizeof(valor));
    W_Q = (valor *) malloc(n_heads * D * d_model * sizeof(valor));
    W_K = (valor *) malloc(n_heads * D * d_model * sizeof(valor));
    W_O = (valor *) malloc(d_model * d_model * sizeof(valor));
    
    for (int i = 0; i < n_heads * D * d_model; ++i) {
      W_V[i] = rand_double();
      W_Q[i] = rand_double();
      W_K[i] = rand_double();
      W_O[i] = rand_double();
    }
  }

  void update_weights(){
    for (int i = 0; i < n_heads * D * d_model; ++i) {
      W_V[i].data += -0.05 * grad[W_V[i].id];
      W_Q[i].data += -0.05 * grad[W_V[i].id];
      W_K[i].data += -0.05 * grad[W_V[i].id];
      W_O[i].data += -0.05 * grad[W_V[i].id];
    }
  }

  valor* pass(valor *E) {

    valor *V, *Q, *K;
    V = (valor *) malloc(n_heads * C * D * sizeof(valor));
    Q = (valor *) malloc(n_heads * C * D * sizeof(valor));
    K = (valor *) malloc(n_heads * C * D * sizeof(valor));

    projecao_linear(W_V, E, V);
    projecao_linear(W_Q, E, Q);
    projecao_linear(W_K, E, K);

    printf("V:\n");
    for (int i = 0; i < n_heads; ++i) {
      printf("Head %d:\n", i);
      for (int j = 0; j < C; ++j) {
        for (int k = 0; k < D; ++k) {
          printf("%lf ", V[(i * C * D) + (j * D) + k].data);
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
          printf("%lf ", Q[(i * C * D) + (j * D) + k].data);
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
          printf("%lf ", K[(i * C * D) + (j * D) + k].data);
        } 
        printf("\n");
      }
    }
    printf("\n");

    // Transpõe a matriz K
    valor *K_transposto = (valor *) malloc(n_heads * D * C * sizeof(valor));
    transpor(K, K_transposto);
    
    printf("K transposto:\n");
    for (int h = 0; h < n_heads; ++h) {
      printf("Head %d:\n", h);
      for (int i = 0; i < D; ++i) {
        for (int j = 0; j < C; ++j) {
          printf("%lf ", K_transposto[(h * D * C) + (i * C) + j].data);
        }
        printf("\n");
      }
    }
    printf("\n");
    
    // Multiplicação Q por K^T
    valor *H = (valor *) malloc(n_heads * C * C * sizeof(valor));
    produto_interno(Q, K_transposto, H);

    printf("H:\n");
    for (int h = 0; h < n_heads; ++h) {
      for (int i = 0; i < C; ++i) {
        for (int j = 0; j < C; ++j) {
          printf("%lf ", H[(h * C * C) + (i * C) + j].data);
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
          printf("%lf ", H[(h * C * C) + (i * C) + j].data);
        }
        printf("\n");
      }
    }
    printf("\n");

    // Obtém Attention em cada head 
    valor *A = (valor *) malloc(n_heads * C * D * sizeof(valor));
    matmul(H, V, A);

    printf("Device A:\n");
    for (int h = 0; h < n_heads; ++h) {
      for (int i = 0; i < C; ++i) {
        for (int j = 0; j < D; ++j) {
          printf("%lf ", A[(h * C * D) + (i * D) + j].data);
        }
        printf("\n");
      }
    }
    printf("\n");

    // Concatena tudo em concat
    valor *A_concat = (valor *) malloc(C * d_model * sizeof(valor));
    concat(A, A_concat);

      
    printf("Concat:\n");
    for (int i = 0; i < C; ++i) {
      for (int j = 0; j < d_model; ++j) {
        printf("%lf ", A_concat[(i * d_model) + j].data);
      }
      printf("\n");
    }
    printf("\n");

    // Calcula o resultado final de multihead
    valor *multihead = (valor *) malloc(C * d_model * sizeof(valor)); 
    projecao_final(A_concat, W_O, multihead);

    return multihead;
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

  valor erro = 0;

  for (int i = 0; i < C; ++i) {
    for (int j = 0; j < d_model; ++j) {
      erro = erro + res[(i * d_model) + j];
      printf("%lf ", res[(i * d_model) + j].data);
    }
    printf("\n");
  }

  cout << erro << '\n';

  valor erro1 = erro;

  erro.backward_pass();

  
  mha.update_weights();

  res =  mha.pass(E);

  erro = 0;

  for (int i = 0; i < C; ++i) {
    for (int j = 0; j < d_model; ++j) {
      erro = erro + res[(i * d_model) + j];
      printf("%lf ", res[(i * d_model) + j].data);
    }
    printf("\n");
  }

  cout << "Taxa de erro: " << erro1 << '\n';
  cout << "Depois de aprender com os erros...\n";
  cout << "Taxa de erro: " << erro << '\n';

  cout << "Total de valores calculados: " << size(valores) << '\n';
}

