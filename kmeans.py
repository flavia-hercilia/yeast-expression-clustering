import numpy as np
import pandas as pd
import os
import io
import matplotlib.pyplot as plt
import gzip
import warnings

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
def calcular_distancia_euclidiana(X, Y):
    """
        Calcula a Distância Euclidiana entre todos os pontos de X e Y,
        ignorando valores ausentes (NaN) e dividindo pelo número dinâmico d
        de atributos válidos em comum.
        Retorna matriz de distâncias (len(X), len(Y)).
    """
    #prepara uma tabela vazia pra guardar as distancias preenchida por 0s 
    n_x = X.shape[0] #numero de linhas => tabela de genes (total de genes)
    n_y = Y.shape[0] #numero de centroides => a quantidade de k quando rodarmos o programa
    dists = np.zeros((n_x, n_y))

    for j in range(n_y):
        #diferença ao quadrado entre todos os genes (X) e o centroide j (Y[j])
        #onde tiver NaN em X ou em Y[j], o resultado fica NaN
        diff_sq = (X - Y[j]) ** 2

        #conta quantas posições válidas existem em comum em cada linha (d)
        d = np.sum(~np.isnan(diff_sq), axis=1)

        #soma as diferenças ao quadrado ignorando os NaNs
        soma_sq = np.nansum(diff_sq, axis=1)

        #divide por d e tira a raiz; se d == 0 (nenhum dado em comum), distância = infinito
        with np.errstate(divide="ignore", invalid="ignore"):
            dist_j = np.where(d > 0, np.sqrt(soma_sq / d), np.inf)

        dists[:, j] = dist_j

    return dists

#------------------------------------------------------------------------------------------------

# CLASSE KMEANS

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
                
                #Atualização: Recalcula a posição dos centroides ignorando NaNs
                new_centroids = np.zeros_like(centroids)
                for i in range(self.k):
                    #pega todos os pontos que pertencem ao cluster i
                    points = X[labels == i]
                    if len(points) > 0:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", category=RuntimeWarning)
                            new_centroids[i] = np.nanmean(points, axis=0)
                    else:
                        #se cluster vazio, mantém o antigo
                        new_centroids[i] = centroids[i]
                
                #convergencia: Verifica se os centroides mudaram pouco (ignorando NaNs)
                shift = np.nansum((centroids - new_centroids)**2)
                centroids = new_centroids
                
                if shift < self.tol:
                    break
            
            #calcula a inercia (soma dos erros quadráticos) da execução
            dists_finais = calcular_distancia_euclidiana(X, centroids)
            min_dists = np.min(dists_finais, axis=1)
            current_inertia = np.nansum(min_dists**2)
            
            #se melhor resultado ate agora, salva
            if current_inertia < best_inertia:
                best_inertia = current_inertia
                best_centroids = centroids
                best_labels = labels
        
        self.centroids = best_centroids
        self.labels = best_labels
        self.inertia = best_inertia

#------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    caminho = os.path.join(script_dir, "GDS38_full.soft.gz")

    df_bruto = carregar_gds(caminho)
    X, df_limpo, colunas_gsm = preparar_dados(df_bruto)

    total_nans = np.isnan(X).sum()
    percentual_nans = (total_nans / X.size) * 100

    # print(f"Genes válidos após limpeza: {X.shape[0]}")
    # print(f"Pontos temporais (colunas GSM): {X.shape[1]}")
    # print(
    #     f"Total de valores ausentes (NaN): {total_nans} ({percentual_nans:.2f}%)"
    # )
    # print("\nExemplo dos 3 primeiros genes reais:")
    # print(df_limpo[["IDENTIFIER", "Gene symbol", "Gene title"]].head(3))

    print("\n--- Testando K-Means (200 genes, k=5) ---")
    km_teste = KMeans(k=5, max_iters=100, tol=0.01, n_init=1)
    km_teste.fit(X[:200])

    print(f"Inércia do teste: {km_teste.inertia:.4f}")
    clusters, contagens = np.unique(km_teste.labels, return_counts=True)
    for c, qtd in zip(clusters, contagens):
      print(f"  Cluster {c}: {qtd} genes")