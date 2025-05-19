// Vamos fazer um valor que implementa backpropagation [x] 
// Vamos construir um neuron [x]
// Vamos construir um layer [x]
// Vamos construir uma MLP [x]

#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cassert>
#include <iomanip>

using namespace std;
using ld = long double;

struct valor; 
vector<valor> valores;
vector<ld> grad;

struct valor{
  ld data;
  int left_child, right_child;  
  int op;
  int id; 
  ld expoente;

  valor(ld _data = 0, int _left_child = -1, int _right_child = -1, int _op = 0, ld _expoente = 0) {
    this->data = _data;
    this->left_child = _left_child;
    this->right_child = _right_child;
    this->op = _op;
    this->id = (int) size(valores);
    this->expoente = _expoente;
    // Faz a cópia deste valor
    valores.push_back(*this); 
    grad.push_back(0);
  }

  // Mostra em algum outputstream
  friend ostream& operator<<(ostream& os, const valor & v) {
    os << "Valor: " << v.data;
    return os;
  }

  valor operator+(valor b) {
    return valor(data + b.data, id, b.id, 1);
  }
  
  valor operator+(ld b) {
    valor x(b);
    return (*this) + x;
  }

  friend valor operator+(ld x, valor a) {
    return a + x;
  }

  valor operator*(valor b) {
    return valor(data * b.data, id, b.id, 2);
  }

  valor operator*(ld b) {
    valor x(b);
    return (*this) * x;
  }

  friend valor operator*(ld b, valor a) {
    return a * b;
  }
  
  valor operator-() {
    return (*this) * -1;
  }

  valor operator-(valor b) {
   return (*this) + -b;
  }

  friend valor operator-(ld b, valor a) {
    return b + -a;
  }

  valor pow(ld k) {
    return valor(powl(data, k), id, -1, 3, k);
  }

  valor operator/(valor b) {
   return (*this) * b.pow(-1);
  }

  valor operator/(ld b) {
   return (*this) * (1 / b);
  }

  friend valor operator/(ld b, valor a) {
   return b * a.pow(-1);
  }

  valor exp() {
    return valor(powl(2.718281828459045, data), id, -1, 4);
  }

  ld exp(ld x) {
    return powl(2.718281828459045, x);
  }

  valor tanh() {
    ld z = (exp(2 * data) - 1) / (exp(2 * data) + 1);
    return valor(z, id, -1, 5); 
  }

  void prop() {
    if (op == 1) {
      grad[left_child] += grad[id];
      grad[right_child] += grad[id];
    } else if (op == 2) {
      grad[left_child] += valores[right_child].data * grad[id];
      grad[right_child] += valores[left_child].data * grad[id];
    } else if (op == 3) {
      grad[left_child] += expoente * powl(valores[left_child].data, expoente - 1) * grad[id];
    } else if (op == 4) {
      grad[left_child] += data * grad[id];
    } else if (op == 5) {
      grad[left_child] += (1 - data * data) * grad[id];
    }
  }

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

ld rand_double(){
  ld min = -1, max = +1;
  ld range = max - min;
  return (ld) rand() / (RAND_MAX / range) + min;
}

struct neuron {
  // Neuron tem um número de inputs e um 
  int n;
  vector<valor> pesos;
  valor bias;
  neuron(int _n) : n(_n) {
    pesos.resize(n);
    for (int i = 0; i < n; ++i) 
      pesos[i] = valor(rand_double());
    bias = valor(rand_double());
  }

  valor pass(vector<valor> &x) {
    assert((int) x.size() == n);
    valor total = 0;
    for (int i = 0; i < n; ++i) {
      total = total + x[i] * pesos[i];
    } 
    total = (total + bias).tanh();
    return total;
  }
};

struct layer {
  vector<neuron> neurons;
  int sz;
  int input_n;

  layer(int _input_n, int _sz) {
    input_n = _input_n;
    sz = _sz;
    for (int i = 0; i < sz; ++i) {
      neurons.push_back(neuron(input_n));
    }
  }

  vector<valor> pass(vector<valor> & x) {
    assert((int) size(x) == input_n);
    vector<valor> ret;
    for (int i = 0; i < sz; ++i) {
      ret.push_back(neurons[i].pass(x));
    }
    return ret;
  }
};

struct MLP{
  vector<layer> layers;
  int n_input = 0;
  MLP(vector<int> tamanhos) {
    n_input = tamanhos[0];
    for (int i = 0; i < (int) tamanhos.size() - 1; ++i) {
      layers.push_back(layer(tamanhos[i], tamanhos[i + 1]));
    }
  }

  vector<valor> pass(vector<valor> & x) {
    assert((int) size(x) == n_input);
    vector<valor> res = x;
    for (layer & l : layers) {
      res = l.pass(res);
    }
    return res;
  }
};

int main(){
  cout << fixed << setprecision(6);
  {
  cout << "Teste 1:\n";

  valor x1(2.0);
  valor x2(0.0);

  valor w1(-3.0);
  valor w2(1.0);

  valor b(6.8813735870195432);

  valor n = x1*w1 + x2*w2 + b;

  valor e = (2*n).exp();
  
  valor o = (e - 1) / (e + 1);

  o.backward_pass();
  
  cout << grad[w1.id] << ' ' << grad[w2.id] << ' ' << grad[b.id] << '\n';
  }

  { 
  cout << "Teste 2:\n";
  valor a(-2);
  valor b(3);
  valor c = a + b;
  valor d = a * b;
  valor e = c * d;
  
  e.backward_pass();

  cout << grad[a.id] << ' ' << grad[b.id] << '\n';
  }
  {
    cout << "Teste 3:\n";
    valor a(1.0);
    valor b = a + a + a + a + a;
    b.backward_pass();
    cout << grad[a.id] << '\n';
  }
  {
    cout << "Rede neural:\n";
    vector<int> tamamhos = {3, 4, 4, 1};
    MLP mlp(tamamhos);

    vector<vector<valor>> entradas = {
      {2.0, 3.0, -1.0},
      {3.0, -1.0, 0.5},
      {0.5, 1.0, 1.0},
      {1.0, 1.0, -1.0}
    };

    vector<valor> valores_esperados = {1.0, -1.0, -1.0, 1.0};

    vector<valor> valores_obtidos;

    cout << "Valores esperados:\n";
    for (int i = 0; i < 4; ++i) {
      cout << valores_esperados[i].data << ' ';
    }
    cout << '\n';

    valor custo = 0;
    // Vamos fazer aqui os treinamentos
    for (int _ = 0; _ < 301; ++_) {
      valores_obtidos.clear();
      for (vector<valor> & entrada : entradas) {
        valores_obtidos.push_back(mlp.pass(entrada)[0]);
      }     

      custo = 0;
      for (int i = 0; i < 4; ++i) {
        custo = custo + (valores_esperados[i] - valores_obtidos[i]).pow(2);
      }

      if (_ == 0) {
        cout << "Valores obtidos no primeiro teste:\n";
        for (int i = 0; i < 4; ++i) {
          cout << valores_obtidos[i].data << ' '; 
        }
        cout << '\n';
        cout << "Cálculo do erro: " << custo.data << '\n';
      }

      for (layer & l : mlp.layers) {
        for (neuron & n : l.neurons) {
          for (valor & w : n.pesos) {
            grad[w.id] = 0.25 * grad[w.id];
          }
        }
      }

      custo.backward_pass();

      for (layer & l : mlp.layers) {
        for (neuron & n : l.neurons) {
          for (valor & w : n.pesos) {
            w.data += -0.10 * grad[w.id];
          }
        }
      }
    }

    cout << "Valores obtidos após 300 passos de treinamento:\n";
    for (int i = 0; i < 4; ++i) {
      cout << valores_obtidos[i].data << ' ';
    }
    cout << '\n';
    cout << "Cálculo do erro: " << custo.data << '\n';
  }
  cout << "Valores totais guardados na memória: " << size(valores) << '\n';
}
