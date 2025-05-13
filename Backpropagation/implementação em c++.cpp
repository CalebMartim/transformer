#include <iostream>
#include <vector>
using namespace std;

template <typename T> 
class Value {
  public:
    T data;
    vector<Value*> _prev;
    string _op;

    Value (T data, vector<Value*> _children = {}, string _op = "") {
      this->data = data;
      this->_prev = _children;
      this->_op = _op;
    }

    // Soma
    Value operator+(Value x) {
      return Value(data + x.data, {this, &x}, "+");
    }

    // Produto
    Value operator*(Value x) {
      return Value(data * x.data, {this, &x}, "*");
    }
    
    // Output stream
    friend ostream& operator<<(ostream& os, const Value & v) {
      os << v.data;
      return os;
    }

  private:
};

int main() { 
  Value<float> a(2);
  Value<float> b (3.0);
  Value<float> c = a + b;
  
  for (Value<float> *x : c._prev) {
    cout << x << ": " << *x << '\n';
  }

  cout << c._op << '\n';

  return 0; 
}