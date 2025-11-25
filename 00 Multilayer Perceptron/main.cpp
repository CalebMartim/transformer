#include <bits/stdc++.h>

using f64 = double;
template <typename T> using V = std::vector<T>;

std::mt19937 rng(std::chrono::steady_clock::now().time_since_epoch().count());
std::uniform_real_distribution<f64> dist(-1, 1);

class Linear{
  private:
    V<V<f64>> weights;
    V<V<f64>> weights_g; // terminação em "_g" refere-se a gradientes
    V<f64> bias;
    V<f64> bias_g;

  public:
    Linear(int in_features, int out_features) {
      weights.resize(in_features, V<f64>(out_features));
      weights_g.resize(in_features, V<f64>(out_features));
      bias.resize(out_features);
      bias_g.resize(out_features);

      for (int i = 0; i < in_features; ++i)
        for (int j = 0; j < out_features; ++j)
          weights[i][j] = dist(rng);
      for (int i = 0; i < out_features; ++i)
        bias[i] = dist(rng);
    }
    Linear(){}
};

class MLP{
  private:
    V<Linear> layers;
  public:
    MLP(int in_features,
    int n_hidden_layers,
    int hidden_dim,
    int out_features) {
      layers.resize(n_hidden_layers + 2);

      layers[0] = Linear(in_features, hidden_dim);
      for (int i = 1; i <= n_hidden_layers; ++i)
        layers[i] = Linear(hidden_dim, hidden_dim);
      layers[n_hidden_layers + 1] = Linear(hidden_dim, out_features);
    }
};

int main(){

  return 0;
}
