import os
import numpy as np
import matplotlib.pyplot as plt
from kmeans import (
    carregar_gds,
    preparar_dados,
    calcular_qualidade_intracluster,
)


def selecionar_top_clusters(X, labels, metrica, top_n=5, min_genes=5):
    """
    Filtra clusters com pelo menos `min_genes` e retorna os `top_n`
    com menor distancia media intra-cluster (mais coesos)
    """
    _, qual_por_cluster = calcular_qualidade_intracluster(X, labels, metrica=metrica)

    candidatos = []
    for cluster_id, dist_media in qual_por_cluster.items():
        qtd_genes = int(np.sum(labels == cluster_id))
        if qtd_genes >= min_genes:
            candidatos.append((cluster_id, dist_media, qtd_genes))

    #ordena da menor distancia (melhor qualidade) para a maior
    candidatos.sort(key=lambda item: item[1])
    return candidatos[:top_n]


def plotar_top_clusters(X, labels, centroids, top_clusters, metrica, caminho_png):
    """
    Gera um painel 2x3 com os perfis temporais dos 5 melhores clusters
    (genes em transparência ao fundo + centroide em destaque).
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    eixo_x = np.arange(1, X.shape[1] + 1)

    for idx, (cluster_id, dist_media, qtd_genes) in enumerate(top_clusters):
        ax = axes[idx]
        pontos = X[labels == cluster_id]
        centroide = centroids[cluster_id]

        #plota cada gene individual do cluster (linhas finas e suaves)
        for gene in pontos:
            ax.plot(eixo_x, gene, color="steelblue", alpha=0.3, linewidth=1.0)

        #plota o centroide em destaque
        ax.plot(
            eixo_x,
            centroide,
            color="crimson",
            linewidth=2.5,
            marker="o",
            markersize=4,
            label="Centroide",
        )

        ax.set_title(
            f"#{idx + 1} | Cluster {cluster_id} ({qtd_genes} genes)\n"
            f"Dist. média: {dist_media:.4f}",
            fontsize=11,
            fontweight="bold",
        )
        ax.set_xlabel("Ponto Temporal (GSM)")
        ax.set_ylabel("Expressão Gênica")
        ax.set_xticks(eixo_x)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(loc="upper right", fontsize=8)

    #esconde o sexto quadrante vazio da grade 2x3
    axes[5].axis("off")

    titulo_metrica = (
        "Correlação de Pearson" if metrica == "pearson" else "Distância Euclidiana Normalizada"
    )
    fig.suptitle(
        f"Top 5 Clusters Mais Coesos — {titulo_metrica} (k=100, ≥ 5 genes)",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(caminho_png, dpi=300)
    plt.close()
    print(f"Gráfico salvo em: {caminho_png}")


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    caminho_gds = os.path.join(script_dir, "GDS38_full.soft.gz")
    caminho_npz = os.path.join(script_dir, "resultados_k100.npz")

    #carrega os dados originais e os resultados salvos em "resultados_k100.npz"
    df_bruto = carregar_gds(caminho_gds)
    X, df_limpo, colunas_gsm = preparar_dados(df_bruto)
    dados = np.load(caminho_npz)

    labels_euc = dados["labels_euc"]
    centroids_euc = dados["centroids_euc"]
    labels_pea = dados["labels_pea"]
    centroids_pea = dados["centroids_pea"]

    #seleciona os 5 melhores clusters de cada metrica
    top5_euc = selecionar_top_clusters(X, labels_euc, metrica="euclidiana")
    top5_pea = selecionar_top_clusters(X, labels_pea, metrica="pearson")

    print("\n--- TOP 5 CLUSTERS (EUCLIDIANA) ---")
    for rank, (cid, dist, qtd) in enumerate(top5_euc, 1):
        print(f"  {rank}º | Cluster {cid:2d} | {qtd:3d} genes | Dist. média: {dist:.4f}")

    print("\n--- TOP 5 CLUSTERS (PEARSON) ---")
    for rank, (cid, dist, qtd) in enumerate(top5_pea, 1):
        print(f"  {rank}º | Cluster {cid:2d} | {qtd:3d} genes | Dist. média: {dist:.4f}")

    #gera e salva os graficos
    print()
    plotar_top_clusters(
        X,
        labels_euc,
        centroids_euc,
        top5_euc,
        metrica="euclidiana",
        caminho_png=os.path.join(script_dir, "top5_euclidiana.png"),
    )
    plotar_top_clusters(
        X,
        labels_pea,
        centroids_pea,
        top5_pea,
        metrica="pearson",
        caminho_png=os.path.join(script_dir, "top5_pearson.png"),
    )