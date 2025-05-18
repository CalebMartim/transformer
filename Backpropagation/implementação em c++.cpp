// Vamos fazer um value que implementa backpropagation [x] 
// Vamos construir um neuron
// Vamos construir um layer
// Vamos construir uma MLP

#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>

using namespace std;
using ld = long double;

struct value; 
vector<value> values;
vector<ld> grad;

struct value{
  ld data;
  int left_child, right_child;  
  int op;
  int id; 
  ld expoente;

  value(ld _data = 0, int _left_child = -1, int _right_child = -1, int _op = 0, ld _expoente = 0) {
    this->data = _data;
    this->left_child = _left_child;
    this->right_child = _right_child;
    this->op = _op;
    this->id = (int) size(values);
    this->expoente = _expoente;
    // Faz a cópia deste valor
    values.push_back(*this); 
    grad.push_back(0);
  }

  // Mostra em algum outputstream
  friend ostream& operator<<(ostream& os, const value & v) {
    os << "Valor: " << v.data;
    return os;
  }

  value operator+(value b) {
    return value(data + b.data, id, b.id, 1);
  }
  
  value operator+(ld b) {
    value x(b);
    return (*this) + x;
  }

  friend value operator+(ld x, value a) {
    return a + x;
  }

  value operator*(value b) {
    return value(data * b.data, id, b.id, 2);
  }

  value operator*(ld b) {
    value x(b);
    return (*this) * x;
  }

  friend value operator*(ld b, value a) {
    return a * b;
  }
  
  value operator-() {
    return (*this) * -1;
  }

  value operator-(value b) {
   return (*this) + -b;
  }

  friend value operator-(ld b, value a) {
    return b + -a;
  }

  value pow(ld k) {
    return value(powl(data, k), id, -1, 3, k);
  }

  value operator/(value b) {
   return (*this) * b.pow(-1);
  }

  value operator/(ld b) {
   return (*this) * (1 / b);
  }

  friend value operator/(ld b, value a) {
   return b * a.pow(-1);
  }

  value exp() {
    return value(powl(2.718281828459045, data), id, -1, 4);
  }

  void prop() {
    if (op == 1) {
      grad[left_child] += grad[id];
      grad[right_child] += grad[id];
    } else if (op == 2) {
      grad[left_child] += values[right_child].data * grad[id];
      grad[right_child] += values[left_child].data * grad[id];
    } else if (op == 3) {
      grad[left_child] += expoente * powl(values[left_child].data, expoente - 1) * grad[id];
    } else if (op == 4) {
      grad[left_child] += data * grad[id];
    }
  }

  void backward_pass(){
    grad[id] = 1;
    
    // Vamos fazer aqui a ordenção topológica:
    vector<int> top_sort;
    vector<bool> vis(size(values));
    auto dfs = [&](int v, auto &&self) -> void {
      if (v == -1 or vis[v]) return;
      vis[v] = true;
      if(values[v].left_child) {
        self(values[v].left_child, self);
        if (values[v].right_child) {
          self(values[v].right_child, self);
        }
      }
      top_sort.push_back(v);
    };
    
    dfs(id, dfs);
    reverse(top_sort.begin(), top_sort.end());

    for (int v : top_sort) {
      values[v].prop();
    }
  }
};

int main(){
  cout << "Teste 1:\n";

  value x1(2.0);
  value x2(0.0);

  value w1(-3.0);
  value w2(1.0);

  value b(6.8813735870195432);

  value n = x1*w1 + x2*w2 + b;

  value e = (2*n).exp();
  
  value o = (e - 1) / (e + 1);

  o.backward_pass();
  
  cout << grad[w1.id] << ' ' << grad[w2.id] << ' ' << grad[b.id] << '\n';

  cout << "Teste 2:\n";
  value teste1(-2);
  value teste2(3);
  value c = teste1 + teste2;
  value d = teste1 * teste2;
  value E = c * d;
  
  E.backward_pass();

  cout << grad[teste1.id] << ' ' << grad[teste2.id] << '\n';
}
