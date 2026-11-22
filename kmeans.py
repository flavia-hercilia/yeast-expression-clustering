import numpy as np
import pandas as pd
import os
import io
import matplotlib.pyplot as plt
import gzip

#------------------------------------------------------------------------------------------------

# CARREGAMENTO E PREPARAÇÃO DE DADOS 
def carregar_gds(caminho_arquivo):
  
  #Lê a tabela de expressão genica do arquivo .soft.gz do NCBI GEO
  linhas_tabela = []
  dentro_da_tabela = False

  #abre o arquivo .gz direto, sem precisar descompactar
  with gzip.open(caminho_arquivo, "rt", encoding="utf-8", errors="ignore") as f:
    for linha in f:
      if linha.startswith("!dataset_table_begin"):
        dentro_da_tabela = True
        continue
      if linha.startswith("!dataset_table_end"):
        break
      if dentro_da_tabela:
        linhas_tabela.append(linha)

  #transforma linhas lidas em um DataFrame do pandas (separado por TAB)
  texto_tabela = "".join(linhas_tabela)
  df = pd.read_csv(io.StringIO(texto_tabela), sep="\t", na_values=["null"])
  return df

def preparar_dados(df):
  #Remove controles de laboratorio e separa a matriz numérica (X) dos metadados

  #mantem apenas linhas que têm Gene title preenchido (remove controle como POLY A)
  df_limpo = df.dropna(subset=["Gene title"]).copy()

  #apenas as colunas de expressão (a 16 que começam com 'GSM')
  colunas_gsm = [col for col in df_limpo.columns if col.startswith("GSM")]

  #remove genes que tenham TODAS as 16 medições vazias (NaN)
  df_limpo = df_limpo.dropna(subset=colunas_gsm, how="all").reset_index(
      drop=True
  )

  #extrai a matriz numérica como array do NumPy (float)
  X = df_limpo[colunas_gsm].values.astype(float)

  return X, df_limpo, colunas_gsm

#------------------------------------------------------------------------------------------------

# DISTANCIA EUCLIDIANA 
#TODO ajustar para ignorar NaN e dividir pelo numero dinamico d
def calcular_distancia_euclidiana(X, Y):
    """
    Calcula a distância euclidiana de todos os pontos de X para todos de Y
    sem usar loops lentos (vetorizado).
    Retorna matriz de distâncias (len(X), len(Y)).
    """
    # Expansão matemática: (a-b)^2 = a^2 + b^2 - 2ab
    X2 = np.sum(X**2, axis=1).reshape(-1, 1)
    Y2 = np.sum(Y**2, axis=1)
    
    #matriz de distancias quadráticas
    dist_sq = X2 + Y2 - 2 * np.dot(X, Y.T)
    
    #garante que não haja valores negativos por erro numerico
    dist_sq = np.maximum(dist_sq, 0)
    
    return np.sqrt(dist_sq)

#------------------------------------------------------------------------------------------------

# CLASSE KMEANS

#TODO ajustar para ignorar NaN na hora de atulizar centroides, no shift e na inercia
class KMeans:
    def __init__(self, k, max_iters=100, tol=0.01, n_init=10):
        self.k = k
        self.max_iters = max_iters
        self.tol = tol
        self.n_init = n_init #numero de vezes que o algoritmo será reiniciado
        self.centroids = None
        self.labels = None
        self.inertia = np.inf

    def fit(self, X):
        best_inertia = np.inf
        best_centroids = None
        best_labels = None
        n_samples = X.shape[0]

        #Executa n_init vezes para evitar mínimos locais
        for _ in range(self.n_init):
            #Inicialização: Escolhe k pontos aleatórios como centroides iniciais
            indices = np.random.choice(n_samples, self.k, replace=False)
            centroids = X[indices]
            
            for it in range(self.max_iters):
                #Atribuição: Calcula distância de cada ponto para os centroides
                dists = calcular_distancia_euclidiana(X, centroids)
                #o rótulo é o índice do centroide com menor distância
                labels = np.argmin(dists, axis=1)
                
                #Atualização: Recalcula a posição dos centroides
                new_centroids = np.zeros_like(centroids)
                for i in range(self.k):
                    #pega todos os pontos que pertencem ao cluster i
                    points = X[labels == i]
                    if len(points) > 0:
                        new_centroids[i] = np.mean(points, axis=0)
                    else:
                        #se cluster vazio, mantém o antigo
                        new_centroids[i] = centroids[i]
                
                #convergencia: Verifica se os centroides mudaram pouco
                shift = np.sum((centroids - new_centroids)**2)
                centroids = new_centroids
                
                if shift < self.tol:
                    break
            
            #calcula a inercia (soma dos erros quadráticos) da execução
            dists_finais = calcular_distancia_euclidiana(X, centroids)
            min_dists = np.min(dists_finais, axis=1)
            current_inertia = np.sum(min_dists**2)
            
            #se melhor resultado ate agora, salva
            if current_inertia < best_inertia:
                best_inertia = current_inertia
                best_centroids = centroids
                best_labels = labels
        
        self.centroids = best_centroids
        self.labels = best_labels

#------------------------------------------------------------------------------------------------

if __name__ == "__main__":
  script_dir = os.path.dirname(os.path.abspath(__file__))
  caminho = os.path.join(script_dir, "GDS38_full.soft.gz")

  df_bruto = carregar_gds(caminho)
  X, df_limpo, colunas_gsm = preparar_dados(df_bruto)

  total_nans = np.isnan(X).sum()
  percentual_nans = (total_nans / X.size) * 100

  print(f"Genes válidos após limpeza: {X.shape[0]}")
  print(f"Pontos temporais (colunas GSM): {X.shape[1]}")
  print(
      f"Total de valores ausentes (NaN): {total_nans} ({percentual_nans:.2f}%)"
  )
  print("\nExemplo dos 3 primeiros genes reais:")
  print(df_limpo[["IDENTIFIER", "Gene symbol", "Gene title"]].head(3))