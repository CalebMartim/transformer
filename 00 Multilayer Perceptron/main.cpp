/*
  TODO:
  - Escrever um backward pass
  - Considerar criar uma classe tensor
  - Fazer um método de loop de treino para MLP
*/


#include <bits/stdc++.h>
using namespace std;

using f64 = double;
template <typename T> using V = vector<T>;

mt19937 rng(chrono::steady_clock::now().time_since_epoch().count());
uniform_real_distribution<f64> dist(-1, 1);

f64 get_rand(){
  return dist(rng);
}

class Linear{
  private:
    V<V<f64>> weights;
    V<V<f64>> weights_g; // terminação em "_g" refere-se a gradientes 
    V<f64> bias;
    V<f64> bias_g; 
    size_t in_features;
    size_t out_features;

  public:
    Linear(){}

    Linear(const size_t _in_features, const size_t _out_features) {
      in_features = _in_features;
      out_features = _out_features;

      weights.resize(in_features, V<f64>(out_features));
      weights_g.resize(in_features, V<f64>(out_features));
      bias.resize(out_features);
      bias_g.resize(out_features);

      
      for (size_t i = 0; i < in_features; ++i) 
        for (size_t j = 0; j < out_features; ++j) 
          weights[i][j] = get_rand();
        
      for (size_t i = 0; i < out_features; ++i)
        bias[i] = get_rand();
    }

    V<f64> dot(V<f64> &x){
      assert(x.size() == in_features);

      V<f64> ret(out_features);

      for (size_t i = 0; i < out_features; ++i) {
        f64 sum = 0.0;

        for (size_t j = 0; j < in_features; ++j) 
          sum += x[j] * weights[j][i];

        ret[i] = sum + bias[i];
      }

      return ret;
    }
};

class MLP{
  private:
    V<Linear> layers;
    size_t n_layers;
    size_t n_parameters;
  public:
    MLP(const size_t in_features, 
        const size_t n_hidden_layers, 
        const size_t hidden_dim, 
        const size_t out_features) {
      n_layers = n_hidden_layers + 2;
      layers.resize(n_layers);
      n_parameters = 0;

      layers[0] = Linear(in_features, hidden_dim);
      n_parameters += in_features * hidden_dim + hidden_dim;

      for (size_t i = 1; i <= n_hidden_layers; ++i) {
        layers[i] = Linear(hidden_dim, hidden_dim);
        n_parameters += hidden_dim * hidden_dim + hidden_dim;
      }

      layers[n_layers - 1] = Linear(hidden_dim, out_features);
      n_parameters += hidden_dim * out_features + out_features;
    }

    V<f64> forward(V<f64> x) {
      for (size_t i = 0; i < n_layers; ++i) 
        x = layers[i].dot(x);

      return x;
    }

    size_t parameters_count(){
      return n_parameters; 
    }
};

int main(){
  const size_t n_input = 100;
  const size_t n_output = 100;

  MLP myMLP(n_input, 100, 100, n_output);

  V<f64> x(n_input);
  for (size_t i = 0; i < n_input; ++i) 
    x[i] = get_rand();

  V<f64> y = myMLP.forward(x);

  for (size_t i = 0; i < n_output; ++i) 
    cout << y[i] << " \n"[i == n_output - 1];
  
  cout << "Number of parameters: " << myMLP.parameters_count() << '\n';

  return 0;
}
