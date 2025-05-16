// Vamos fazer um value que implementa backpropagation 
// Vamos construir um neuron
// Vamos construir um layer
// Vamos construir uma MLP

#include <iostream>
#include <vector>
#include <queue>
#include <cmath>

using namespace std;
using ld = long double;

struct value; 
vector<value*> values;

struct value{
  ld data, grad;
  value *left_child, *right_child;  
  string op;
  int id; 
  ld expoente;

  value(ld data = 0, value *left_child = nullptr, value *right_child = nullptr, string op = "", ld expoente = 0){
    this->data = data;
    this->grad = 0;
    this->left_child = left_child;
    this->right_child = right_child;
    this->op = op;
    this->id = size(values);
    this->expoente = expoente;
    values.push_back(this);
  }

  // Mostra em algum outputstream
  friend ostream& operator<<(ostream& os, const value & v) {
    os << "Valor: " << v.data;
    return os;
  }

  value operator+(value b) {
    return value(this->data + b.data, this, values[b.id], "+");
  }
  
  value operator+(ld b) {
    return value(this->data + b, this, nullptr, "+");
  }

  friend value operator+(ld x, value a) {
    return value(a.data + x, &a, nullptr, "+");
  }

  value operator*(value b) {
    return value(this->data * b.data, this, values[b.id], "*");
  }

  value operator*(ld b) {
    return value(this->data * b, this, nullptr, "*");
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
    return value(powl(this->data, k), this, nullptr, "pow", k);
  }

  value operator/(value b) {
    return (*this) * b.pow(-1);
  }

  friend value operator/(ld b, value a) {
    return a.pow(-1) * b;
  }

  value exp() {
    return value(powl(2.718281828459045, this->data), this, nullptr, "exp");
  }

  void prop() {
    string op = this->op;
    if (op == "+") {
      this->left_child->grad += this->grad;
      this->right_child->grad += this->grad;
    } else if (op == "*") {
      this->left_child->grad += this->right_child->data * this->grad;
      this->right_child->grad += this->left_child->data * this->grad;
    } else if (op == "exp") {
      this->left_child->grad += this->left_child->data * this->grad;
    } else if (op == "pow") {
      this->left_child->grad += this->expoente * powl(this->left_child->data, this->expoente - 1) * this->grad;
    }
  }

  void backward_pass(){
    this->grad = 1;
    this->prop();
  }
};

int main(){
  value a(10.0);
  value b(2.0);
  value d(1.5);
  value c = b + ((2 + (((b + a) + b))) + a) * d;

  // cout << size(values) << '\n';
  // for (int i = 0; i < size(values); ++i) {
  //   cout << *values[i] << '\n';
  // }

  queue<value*> bfs;
  vector<bool> vis(size(values));

  bfs.push(&c);
  while (bfs.empty() == false) {
    value &x = *bfs.front(); 
    bfs.pop();
    if (vis[x.id]) continue;
    vis[x.id] = true;
    cout << x << ' ' << x.op << '\n';
    if (x.left_child != nullptr) {
      bfs.push(x.left_child);
      if (x.right_child != nullptr) {
        bfs.push(x.right_child);
      }
    }
  } 

  // cout << "Mudança de valores\n";
  // cout << *values[0] << '\n';
  // a.data = 3;
  // cout << *values[0] << '\n';

  cout << '\n' << "Teste de gradiente:\n";

  value x(3.0);
  value y(5.0);
  cout << "x: " << x << '\n' << "y: " <<  y << '\n';
  value z = x * y;
  cout << "Resultado " << z << '\n';

  z.backward_pass();

  cout << "Gradientes:\n";
  cout << "x: " << x.grad << '\n';
  cout << "y: " << y.grad << '\n';
  cout << "z: " << z.grad << '\n';

  
}