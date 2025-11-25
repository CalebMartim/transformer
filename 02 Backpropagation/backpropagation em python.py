import random

# Calcula e^x
def exp(x):
    return 2.718281828459045 ** x

# Classe que permite implementarmos o algoritmo
# de backpropagation em valores utilizados 
# em nossas redes neurais
class Value:
  def __init__(self, data, children = []):
    # Valor em si da estrutura
    self.data = data

    # Gradiente após passar um backward pass 
    self.grad = 0.0

    # Nós que apontam para esse neurônio 
    self.children = children

    # Função de propagação dependendo da operação principal usada para
    # gerar esse neurônio
    self.prop = lambda: None 

  def __repr__(self):
    return f"{self.data}"

  def __add__(self, other):
    other = other if isinstance(other, Value) else Value(other)
    out = Value(self.data + other.data, [self, other])
    
    # d(a + b)/da = 1
    # d(a + b)/db = 1
    def prop():
      self.grad += out.grad
      other.grad += out.grad
    out.prop = prop

    return out
  
  def __radd__(self, other):
    return self + other

  def __mul__(self, other):
    other = other if isinstance(other, Value) else Value(other)
    out = Value(self.data * other.data, [self, other])
    
    # d(a*b)/da = b
    # d(a*b)/db = a
    def prop():
      self.grad += other.data * out.grad
      other.grad += self.data * out.grad
    out.prop = prop

    return out
  
  def __rmul__(self, other):
    return self * other

  def __neg__(self):
    return self * (-1)

  def __sub__(self, other):
    return self + (-other)

  def __rsub__(self, other):
    return other + (-self)
  
  def __pow__(self, other):
    assert isinstance(other, (int, float)), "Não implementado expor por Value"
    out = Value(self.data ** other, [self])

    # d(x^k)/dx = k * (x ** (k - 1))
    def prop():
      self.grad += (other) * (self.data ** (other - 1)) * out.grad;
    out.prop = prop

    return out

  def __truediv__(self, other):
    return self * (other ** -1)
  
  def __rtruediv__(self, other):
    return other * (self ** -1) 

  def tanh(self):
    x = self.data
    y = (exp(2*x) - 1)/(exp(2*x) + 1)
    out = Value(y, [self])

    # d(tanh(x))/dx = 1 - tanh(x)^2
    def prop():
      self.grad += (1 - out.data ** 2) * out.grad
    out.prop = prop

    return out
  
  def exp(self):
    x = self.data
    y = exp(x)
    out = Value(y, [self])

    # d(e^x)/dx = e^x
    def prop():
      self.grad += y * out.grad
    out.prop = prop

    return out
  
  def backward_pass(self):
    # Derivada de x com relação a x é sempre 1
    self.grad = 1
    
    # Fazemos uma ordenação topológica porque podemos
    # propagar a partir de um nó apenas quando 
    # todos os nós que propagam para ele já fizeram
    # essa propagação. Aqui, implementamos o algoritmo
    # de Kosaraju
    vis = set()
    order = []
    def build(v):
      if v not in vis:
        vis.add(v)
        for ch in v.children:
          build(ch)
        order.append(v)

    build(self)

    for node in reversed(order):
      node.prop()

# Teste de backward pass (1):
# Entrada
# x1 = Value(2.0)
# x2 = Value(0.0)

# Pesos
# w1 = Value(-3.0)
# w2 = Value(1.0)

# Bias
# b = Value(6.8813735870195432)

# Cálculo da soma com pesos
# n = x1 * w1 + x2 * w2  + b

# Função de ativação
# e = (2 * n).exp()

# Output
# o = (e - 1) / (e + 1)
# o.backward_pass()

# Gradiantes dos parâmetros 
# print(w1.grad, w2.grad, b.grad)


# Teste de backward pass (2):
# a = Value(1.0)
# b = a + a
# b.backward_pass()
# print(a.grad)

# Teste de backward pass (3):
# a = Value(-2.0)
# b = Value(3)
# c = a + b 
# d = a * b
# e = c * d
# e.backward_pass()
# print(a.grad, b.grad)

class Neuron:
  def __init__(self, n_in):
    # Pesos iniciais para as arestas ligadas nesse neuron
    self.w = [Value(random.uniform(-1, 1)) for _ in range(n_in)]

    # Bias do neuron
    self.b = Value(random.uniform(-1, 1))

  def __call__(self, entrada):
    # Soma ponderada da entrada e os pesos das arestas de entrada
    act = sum([xi * wi for (xi, wi) in zip(entrada, self.w)], self.b)

    # Função de normalização
    return act.tanh() 

  def parameters(self):
    return self.w + [self.b]

class Layer:
  def __init__(self, n_in, n_neurons):
    self.neurons = [Neuron(n_in) for _ in range(n_neurons)]
  
  def __call__(self, entrada):
    act = [neuron(entrada) for neuron in self.neurons]
    return act[0] if len(act) == 1 else act
  
  def parameters(self):
    return [p for neuron in self.neurons for p in neuron.parameters()]

# MLP = Multilayer Perceptron, junção de cada camada
class MLP:
  def __init__(self, layers_sz):
    self.layers = [Layer(layers_sz[i], layers_sz[i + 1]) for i in range(len(layers_sz) - 1)]

  def __call__(self, entrada):
    saida = entrada
    for layer in self.layers:
      saida = layer(saida)
    return saida

  def parameters(self):
    return [p for layer in self.layers for p in layer.parameters()]

# Número de neurônios em cada camada da rede
layers_sz = [3, 4, 4, 1] 

mlp = MLP(layers_sz)

### Exemplo de treinamento:

# Batch de treinamento
xs = [
  [2.0, 3.0, -1.0],
  [3.0, -1.0, 0.5],
  [0.5, 1.0, 1.0],
  [1.0, 1.0, -1.0]
]

# Saída esperada para cada valor
ys = [1.0, -1.0, -1.0, 1.0] 

# Saída da rede neural para cada treino
ypred = []

for p in mlp.parameters():
  p.data += -0.01 * p.grad

for i in range(301):
  # Aplica cada treinamento do batch à rede neural
  ypred = [mlp(x) for x in xs]

  if (i == 0):
      print(f'Previsões iniciais:\n{ypred}\n')

  # A função de perda, aqui, é definida como a soma do
  # quadrado das diferenças entre os valores desejados 
  # e obtidos
  loss = sum([(Value(ydesejado) - yout)**2 for ydesejado, yout in zip(ys, ypred)], Value(0.0))

  if (i % 50 == 0):
      print(f'Iteração {i}:\nPerda: {loss}')
  
  # Atualiza o gradiente de todos os parâmetros mantendo 
  # um pouco de seus valores anteriores
  for p in mlp.parameters():
      p.grad = 0.25 * p.grad

  loss.backward_pass()
  
  for p in mlp.parameters():
    p.data += -0.1 * p.grad

print(f'\nSaída desejada:\n{ys}')
print(f'\nPrevisões finais:\n{ypred}')

