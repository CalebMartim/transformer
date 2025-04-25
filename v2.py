import torch 
import numpy as np
import torch.nn as nn
from torch.nn import functional as F

# Parâmetros:
B = 32 # Quantas sequências diferentes estarão sendo treinadas em paralelo
context_window = 8 # Tamanho da janela de contexto
iteracoes_treinamento = 5000 # Quantos passos de treinamento ele vai fazer 
intervalo_avaliacao = 300 # intervalo para checar avaliação de treinamneto no treinamneto
taxa_de_aprendizagem = 1e-3 # entendo só mais ou menos o que isso significa
device = 'cuda' if torch.cuda.is_available() else 'cpu' # usa a gpu, se possível
iteracoes_avaliacao = 200 # iterações para pegar a mediana de erro nos intervalos de avaliação
d_model = 32 # tamanho do vetor de embedding de cada token
n_head = 4 # número de cabeças de atenção
n_layer = 4 # número de camadas de atenção
dropout = 0.2 # taxa de dropout (não sei o que é isso ainda)
# -------

# Inserindo uma semente arbitrária para gerar números aleatórios
torch.manual_seed(1337)

# Leitura do arquivo de treino
# input.txt = https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('input.txt', 'r', encoding='utf-8') as f:
    texto = f.read()

# [codificação e decodificação]
# Em LLMs, é necessário codificar sequências 
# de texto em tokens. Isso geralmente é feito 
# com ferramentas como SentencePiece ou tiktoken, que 
# dividem palavras em subpalavras, mas, aqui, estamos 
# usando apenas um mapeamento de caractére para inteiro, 
# de acordo com a posição de cada caractére existente do 
# input na ordem lexicográfica.
vocabulario  = sorted(list(set(texto)))
tamanho_vocabulario = len(vocabulario)
stoi = {} # string para int
itos = {} # int para string
for i, s in enumerate(vocabulario):
    stoi[s] = i
    itos[i] = s
encode = lambda s: [stoi[c] for c in s]
decode = lambda s: ''.join([itos[c] for c in s])

# Transformamos a codificação do nosso input inteiro em um tensor de rank 1.
# (O tensor permite aplicarmos operações como soma de vetores e produto escalar)
dados = torch.tensor(encode(texto), dtype=torch.int64) # dtype significa 'data type'

# Agora, precisamos dividir nosso texto em duas partes, uma 
# que será treinada no modelo e outra que usaremos para avaliar o
# nosso modelo (para garantir que não há 'overfitting'). 
# Vamos dar 90% dos dados para treinamento e 10% para avaliação.
divisao = int(0.9 * len(dados))
dados_para_treinamento = dados[:divisao]
dados_para_validacao = dados[divisao:]

def get_batch(split):
    # Estaremos treinando, em paralelo, B diferentes
    # contextos. Para isso, pegamos B índices aleatórios,
    # criamos uma torch stack desses contextos, cada um, de tamanho context_window.
    # E criamos uma torch stack dos alvos de cada contexto também.
    # (os alvos são os tokens que queremos prever, ou seja, o próximo token de cada contexto)

    dados = dados_para_treinamento if split == 'train' else dados_para_validacao 

    # Pega uma lista de B números aleatórios que variam entre 0 e (len(dados) - context_window - 1):
    indices = torch.randint(len(dados) - context_window, (B,)) 

    # Cria os batches em si, pegamos os B sequências de tokens, cada um começando em i e terminando em (i + context_window - 1):
    # (note que estamos transformando uma lista de listas e transformando-a em uma torch.stack)
    x = torch.stack([dados[i : i + context_window] for i in indices])
    
    # Pegamos os alvos para cada prefixo de cada sequência de x:  
    y = torch.stack([dados[i + 1 : i + context_window + 1] for i in indices])

    # Movendo os parâmetros para device (cuda ou cpu)
    x, y = x.to(device), y.to(device)
      
    return x, y
# Não queremos calcular o gradiente para essa função
# Isso é útil para economizar memória e acelerar o processo
# de avaliação, já que não precisamos atualizar os pesos do modelo.
@torch.no_grad() 
def estimar_perda():
    retorno = {}
    modelo.eval() # entra em modo de avaliação
    for split in ['train', 'val']:
        losses = torch.zeros(iteracoes_avaliacao) # inicializa um tensor de zeros
        for k in range(iteracoes_avaliacao):
            X, Y = get_batch(split) 
            logits, loss = modelo(X, Y)
            losses[k] = loss.item() # pega o valor da perda
        retorno[split] = losses.mean()
    modelo.train() # volta para o modo de treinamento
    return retorno

# Classe que define um head de atenção
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        
        # Camadas lineares que vão criar a partir do vetor de embedding
        # os vetores k, q e v
        self.key  = nn.Linear(d_model, head_size, bias = False)
        self.query = nn.Linear(d_model, head_size, bias = False)
        self.value = nn.Linear(d_model, head_size, bias = False)

        # Cria uma matriz triangular inferior (que não é um parâmetro do modelo, por isso usamos register_buffer)
        self.register_buffer('tril', torch.tril(torch.ones(context_window, context_window))) 
    
        self.dropout = nn.Dropout(dropout)
    # Sobreescreve nn.Module e roda quando chamamos um objeto da classe com esses parâmetros 
    def forward(self, x):
        # x são os batches que contém os contextos que contém os embeddings dos tokens
        B, T, C = x.shape

        # Aplica x nas camadas lineares, criando os vetores de chave, consulta e valor 
        # a partir dos embeddings dos tokens 
        k = self.key(x) # shape = (B, T, head_size)
        q = self.query(x) 
        v = self.value(x)

        # Calcula o produto escalar entre as consultas e as chaves,
        # fazendo uma transposição para alinhar as dimensões 
        # e normalizando os valores pelo inverso da raiz quadrada do tamanho da cabeça.
        # Isto aqui que é o \frac{QK^T}{sqrt(d_k)} 
        wei = q @ k.transpose(-2, -1) * (C ** -0.5) # shape = (B, T, head_size) @ (B, head_size, T) = (B, T, T)
        
        # Aplica a máscara triangular inferior
        # Isso garante que cada token só possa ver os tokens anteriores.
        # Também significa que isso se trata do decoder do transformer
        # (o encoder não tem essa máscara, porque ele não precisa prever nada)
        wei = wei.masked_fill(self.tril[:T,:T] == 0, float('-inf'))
        
        # Aplica softmax na última dimensão para normalizar os pesos em probabilidades
        wei = F.softmax(wei, dim = -1)

        wei = self.dropout(wei) 
        
        # Multiplica os pesos pelos valores
        # Isso aqui finaliza a fórmula:
        # softmax(\frac{QK^T}{sqrt(d_k)})V
        attention = wei @ v 

        return attention # shape = (B, T, head_size)

# multi-head attention serve para treinarmos
# várias cabeças de atenção em paralelo
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        # Cada cabeça de atenção tem um tamanho head_size
        # ModuleList serve para armazenar uma lista de módulos
        # (neste caso, cabeças de atenção)
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])

        # linear layer que vai mapear os embeddings para o tamanho do vocabulário
        # essa projeção serve para 
        self.proj = nn.Linear(d_model, d_model) 

        self.dropout = nn.Dropout(dropout) 
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim = -1) 
        out = self.proj(out)
        return out
        # Rodamos num_heads cabeças de atenção para o mesmo tensor x
        # no final, concatenamos cada uma dos attentions na última dimensão
        # obtendo um resultado de tamanho B x T x (num_heads * head_size)
        # return torch.cat([h(x) for h in self.heads], dim = -1) 
        

# Uma rede neural feedforward (multilayer perceptron)
# expande o vetor de embedding para 4 vezes o tamanho do embedding
# aplica uma função de ativação ReLU
# e depois reduz de volta para o tamanho do embedding.
class FeedForward(nn.Module):
    def __init__(self, d_model):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(0.2) # dropout é uma técnica que ajuda a evitar overfitting
        )
    
    def forward(self, x):
        return self.net(x)
    
class block(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()

        head_size = d_model // n_heads # head_size nos dará os valores d_k, d_q, d_v
        self.sa = MultiHeadAttention(n_heads, head_size) 
        self.ffwd = FeedForward(d_model) 

        # Normalização de camada é uma técnica que ajuda a estabilizar o treinamento
        # e a melhorar a convergência do modelo. Ela normaliza as ativações
        # de cada camada para que tenham média zero e variância um.
        self.ln1 = nn.LayerNorm(d_model) # Normalização de camada
        self.ln2 = nn.LayerNorm(d_model) # Normalização de camada

    def forward(self, x):
        # Passamos nosso input x pelas cabeças de atenção 
        # e somamos o resultado com o input original
        # (isso é chamado de residual connection)
        # Isso ajuda a evitar o problema de gradientes desaparecendo
        x = x + self.sa(self.ln1(x))

        # passamos o input x pela rede neural feedforward
        # para expandir o conhecimento do modelo a partir 
        # de cada token 
        x = x + self.ffwd(self.ln2(x))
        return x

# Esta classe define o modelo de linguagem principal.
# É este o modelo que será treinado e que irá gerar texto.
class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # Aqui, estamos definindo um parâmetro que pode ser aprendível pela função
        # de otimização. Esta é a tabela de embedding que cria um vetor de dimensão
        # d_model para cada possível token. Como temos tamanho_vocabulario tokens,
        # esta tabela tem dimensão (tamanho_vocabulario, d_model)
        self.token_modelling_table = nn.Embedding(tamanho_vocabulario, d_model)

        # Estamos mapeando as posições dos tokens para vetores de embedding
        self.position_modelling_table = nn.Embedding(context_window, d_model) 

        # Para o input x, iremos passá-lo por 3 blocos de atenção.
        # Isso é feito para aumentar a capacidade do modelo de aprender
        # padrões sobre os dados. Cada bloco de atenção tem 4 cabeças de atenção.
        # self.blocks = nn.Sequential(
        #     block(d_model, 4), 
        #     block(d_model, 4),
        #     block(d_model, 4),
        #     nn.LayerNorm(d_model), 
        # )
        self.blocks = nn.Sequential(*[block(d_model, n_head) for _ in range(n_layer)]) # Isso aqui é o mesmo que o de cima, mas mais curto
        self.ln_f = nn.LayerNorm(d_model) # Normalização de camada

        # Linear layer que vai mapear os embeddings para o tamanho do vocabulário
        # (lm_head significa language model head)
        self.lm_head = nn.Linear(d_model, tamanho_vocabulario) # matriz de unembedding
 
    def forward(self, idx, targets = None): 
        B, T = idx.shape # B = tamanho do batch, T = tamanho da janela de contexto

        # Isso aqui está criando um array 
        # tridimensional B x T x C onde 
        # B = tamanho de batches (32)
        # T = tamanho da janela de contexto (8)
        # C = d_model (32)
        # Este array está me dando o embedding
        # do c-ésimo token no t-ésimo contexto 
        # do b-ésimo batch: 
        embedding_dos_tokens = self.token_modelling_table(idx) 
        
        # Isso aqui pega o embedding de cada uma das posições
        embedding_das_posicoes = self.position_modelling_table(torch.arange(T, device = device)) 
        
        # Estamos somando os dois vetores de embedding para cada token em idx (embedding do token e embedding da posição)
        h = embedding_dos_tokens + embedding_das_posicoes # h = hidden states

        # Aplicando os embeddings às cabeças de atenção
        h = self.blocks(h) 
        h = self.ln_f(h) # Normalização de camada

        # Estamos pegando os valores do resultado do bloco de atenção (os hidden states)
        # e transformando os vetores de dimensão d_model (os embeddings de cada token) 
        # em vetores de tamanho tamanho_vocabulario. Ou seja, em cada token da nossa sequência, 
        # estamos atribuindo valores para cada token do vocabulário. Note que h, neste momento, 
        # tem forma (B, T, num_heads * head_size), mas, num_heads * head_size = d_model,
        # por isso, é possível aplicá-la à lm_head. No final, lm_head(h) tem forma 
        # (B, T, tamanho_vocabulario)
        logits = self.lm_head(h) 
        
        if targets == None:
            loss = None
        else:
            # Estamos simplesmente pegando as dimensões do nosso tensor 
            B, T, C = logits.shape
            
            # Estamos re-estruturando nosso tensor para 
            # um tensor bidimensional B*T x C para podermos
            # usar a função cross_entropy do pytorch 
            logits = logits.view(B * T, C)
            
            # Estamos reduzindo também as dimensões
            # dos nossos alvos para podemos mapear os 
            # valores adequadamente em cross_entropy também 
            targets = targets.view(B * T)

            # Função de perda
            # Aplicamos log_softmax a cada lista em logits
            # e vamos calcular o negativo da média dos valores
            # escolhido por targets em cada lista.
            # [explicar melhor depois]
            loss = F.cross_entropy(logits, targets)
        
        return logits, loss

    def generate(self, idx, novos_tokens):
        # idx é um array bidimensional B x T que são os B contextos de tamanho T
        for _ in range(novos_tokens):
            # Pega as previsões para cada prefixo de cada batch 

            idx_cond = idx[:, -context_window:] # Pega os últimos contex_window tokens de cada batch
            logits, loss = self(idx_cond)

            # Um jeito de pegar apenas os logits do último entre os context_window tokens de cada bath 
            logits = logits[:, -1, :] 
                
            # Normaliza os logits em probabilidades 
            # (dim = 1 específica que estamos fazendo isso nos logits de cada batch, não no batch em si)
            probs = F.softmax(logits, dim = 1) 
            
            # Dado a distribuição de probabilidade, escolhe um valor  
            # de cada dimensão seguindo a distribuição. Por exemplo se você tem [0.2, 0.8]
            # você tem 20% de probabilidade de pegar o valor 0 e 
            # 80% de pegar o valor 1
            idx_next = torch.multinomial(probs, num_samples = 1)
            
            # Concatena os novos tokens em cada um dos batches 
            idx = torch.cat((idx, idx_next), dim = 1)

        return idx
    
modelo = BigramLanguageModel()
m = modelo.to(device)

# Vamos criar um otimizador de função 
optimizador = torch.optim.AdamW(modelo.parameters(), lr = taxa_de_aprendizagem) 

for iter in range(iteracoes_treinamento):

    if iter % intervalo_avaliacao == 0:
        losses = estimar_perda()
        print(f'step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}')
    
    # Pega um batch aleatório 
    xb, yb = get_batch('train')

    # Calcula os logits e a função de custo
    logits, loss = m(xb, yb)

    # só Deus sabe
    optimizador.zero_grad(set_to_none = True)
    loss.backward()
    optimizador.step()

# Mostra um exemplo
context = torch.zeros((1, 1), dtype = torch.int64)
print(decode(m.generate(context, novos_tokens = 500)[0].tolist()))
