# Bibliotecas
import math
import numpy as np
from collections import deque

# Equivalente a um neurônio
class Value:
  def __init__(self, data, children = []):
    # Dado do neurônio
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

  def __pow__(self, other):
    assert isinstance(other, (int, float)), "Não implementado expor por Value"
    out = Value(self.data ** other, [self])

    def prop():
      self.grad += (other) * (self.data ** (other - 1)) * out.grad;
    out.prop = prop

    return out

  def __truediv__(self, other):
    out = self * other ** -1
    return out
  
  def __rtruediv__(self, other):
    return other * self ** -1 

  def tanh(self):
    x = self.data
    y = (math.exp(2*x) - 1)/(math.exp(2*x) + 1)
    out = Value(y, [self])

    def prop():
      self.grad += (1 - out.data ** 2) * out.grad
    out.prop = prop

    return out
  
  def exp(self):
    x = self.data
    y = math.exp(x)
    out = Value(y, [self])

    def prop():
      self.grad += y * out.grad
    out.prop = prop

    return out
  
  def backward_pass(self):
    self.grad = 1
    
    # Kosaraju
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


# Entrada
x1 = Value(2.0)
x2 = Value(0.0)

# Pesos
w1 = Value(-3.0)
w2 = Value(1.0)

# Bias
b = Value(6.8813735870195432)

# Cálculo da soma com pesos
n = x1 * w1 + x2 * w2  + b

# Função de ativação
e = (2 * n).exp()
o = (e - 1) / (e + 1)

# Backpropagation
o.backward_pass()

print(w1.grad, w2.grad, b.grad, x1.grad, x2.grad)

a = Value(1.0)
b = a + a
b.backward_pass()
print(a.grad)

a = Value(-2.0)
b = Value(3)
c = a + b 
d = a * b
e = c * d
e.backward_pass()
print(a.grad, b.grad)

import torch
import random

n = 3

class Neuron:
  def __init__(self, nin):
    self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
    self.b = Value(random.uniform(-1, 1))

  def __call__(self, x):
    act = sum([xi * wi for (xi, wi) in zip(x, self.w)], self.b)
    out = act.tanh()
    return out

class Layer:
  def __init__(self, nin, nout):
    self.neurons = [Neuron(nin) for _ in range(nout)]
  
  def __call__(self, x):
    act = [n(x) for n in self.neurons]
    act = [x.tanh() for x in act]
    return act[0] if len(act) == 1 else act
  
class MLP:
  def __init__(self, nin, nouts):
    sz = [nin] + nouts
    self.layers = [Layer(sz[i], sz[i + 1]) for i in range(len(nouts))]

  def __call__(self, x):
    for layer in self.layers:
      x = layer(x)
    return x

nin = 3
nouts = [4, 4, 1]
x = [Value(random.uniform(-1, 1)) for _ in range(n)]
mlp = MLP(nin, nouts)

print(mlp(x))
