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

# DISTANCIA DE PEARSON

def calcular_distancia_pearson(X, Y):
    """
        Calcula a distância baseada na Correlação de Pearson: d_p = 1 - Corr(V, U),
        ignorando valores ausentes (NaN) e usando apenas os atributos válidos em comum
    """
    n_x = X.shape[0]
    n_y = Y.shape[0]
    dists = np.zeros((n_x, n_y))

    for j in range(n_y):
        y = Y[j]
        #mascara booleana: True onde TANTO o gene (X) quanto o centroide (y) não são NaN
        mask = ~np.isnan(X) & ~np.isnan(y)
        d = np.sum(mask, axis=1)

        #calcula a media de X e de y considerando as posições validas em comum
        d_seguro = np.where(d > 0, d, 1)[:, None]
        mean_x = np.sum(np.where(mask, X, 0.0), axis=1, keepdims=True) / d_seguro
        mean_y = np.sum(np.where(mask, y, 0.0), axis=1, keepdims=True) / d_seguro

        #subtrai a média (centraliza) e zera onde havia NaN
        dx = np.where(mask, X - mean_x, 0.0)
        dy = np.where(mask, y - mean_y, 0.0)

        #formula da Correlação de Pearson
        numerador = np.sum(dx * dy, axis=1)
        denominador = np.sqrt(np.sum(dx**2, axis=1) * np.sum(dy**2, axis=1))

        with np.errstate(divide="ignore", invalid="ignore"):
            corr = np.where((d > 1) & (denominador > 0), numerador / denominador, np.nan)
            #garante que erros de arredondamento não deixem corr fora de [-1, 1]
            corr = np.clip(corr, -1.0, 1.0)
            dist_j = np.where(~np.isnan(corr), 1.0 - corr, 1.0)

        dists[:, j] = dist_j

    return dists


#------------------------------------------------------------------------------------------------

# CLASSE KMEANS

class KMeans:
    def __init__(self, k, max_iters=100, tol=0.01, n_init=10, metrica="euclidiana"):
        self.k = k
        self.max_iters = max_iters
        self.tol = tol
        self.n_init = n_init #numero de vezes que o algoritmo será reiniciado
        self.metrica = metrica
        self.centroids = None
        self.labels = None
        self.inertia = np.inf

    def _calcular_distancias(self, X, centroids):
        if self.metrica == "pearson":
            return calcular_distancia_pearson(X, centroids)
        return calcular_distancia_euclidiana(X, centroids)

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
                dists = self._calcular_distancias(X, centroids)
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
            dists_finais = self._calcular_distancias(X, centroids)
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

# QUALIDADE INTRACLUSTER

def calcular_qualidade_intracluster(X, labels, metrica="euclidiana"):
    """
    Calcula a distancia media intra-cluster
    Retorna:
      - media_geral: media das distancias intra-cluster de todos os clusters validos
      - qualidade_por_cluster: dicionário {id_cluster: distancia_media}
    """
    func_dist = (
        calcular_distancia_pearson
        if metrica == "pearson"
        else calcular_distancia_euclidiana
    )
    qualidade_por_cluster = {}

    for c in np.unique(labels):
        pontos = X[labels == c]
        n = len(pontos)

        #só faz sentido calcular distância entre pares se houver pelo menos 2 genes no cluster
        if n >= 2:
            #matriz de distancias de todos contra todos dentro do mesmo cluster
            D = func_dist(pontos, pontos)
            #pega o triangulo superior da matriz (evita contar a diagonal 0 e pares repetidos)
            triangulo_sup = D[np.triu_indices(n, k=1)]
            #filtra apenas distancias finitas (ignora NaN e infinito)
            pares_validos = triangulo_sup[np.isfinite(triangulo_sup)]
            if len(pares_validos) > 0:
                qualidade_por_cluster[c] = np.mean(pares_validos)

    media_geral = np.mean(list(qualidade_por_cluster.values()))
    return media_geral, qualidade_por_cluster

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

    print("\n--- Testando K-Means (200 genes, k=5) ---")
    km_teste = KMeans(k=5, max_iters=100, tol=0.01, n_init=1)
    km_teste.fit(X[:200])

    print(f"Inércia do teste: {km_teste.inertia:.4f}")
    clusters, contagens = np.unique(km_teste.labels, return_counts=True)
    for c, qtd in zip(clusters, contagens):
      print(f"  Cluster {c}: {qtd} genes")

    print("\n--- Testando K-Means Pearson (200 genes, k=5) ---")
    km_pearson = KMeans(k=5, max_iters=100, tol=0.01, n_init=1, metrica="pearson")
    km_pearson.fit(X[:200])

    print(f"Inércia Pearson (teste): {km_pearson.inertia:.4f}")
    clusters_p, contagens_p = np.unique(km_pearson.labels, return_counts=True)
    for c, qtd in zip(clusters_p, contagens_p):
      print(f"  Cluster {c}: {qtd} genes")

    qual_euc, _ = calcular_qualidade_intracluster(
        X[:200], km_teste.labels, metrica="euclidiana"
    )
    qual_pea, _ = calcular_qualidade_intracluster(
        X[:200], km_pearson.labels, metrica="pearson"
    )

    print(f"\nQualidade média intra-cluster (Euclidiana): {qual_euc:.4f}")
    print(f"Qualidade média intra-cluster (Pearson):    {qual_pea:.4f}")