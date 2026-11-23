import os
from collections import Counter
import numpy as np
from kmeans import carregar_gds, preparar_dados
from analise import selecionar_top_clusters


def extrair_top_go(serie_go, top_n=5):
    """
    Conta os termos mais frequentes em uma coluna GO do GEO
    """
    contador = Counter()
    for entrada in serie_go.dropna():
        for termo in str(entrada).split("///"):
            termo_limpo = termo.strip()
            if termo_limpo:
                contador[termo_limpo] += 1
    return contador.most_common(top_n)


def validar_metrica(X, df_limpo, labels, metrica, pasta_saida):
    """
    Extrai e exibe os principais processos biologicos (GO:Process) e
    exporta CSV/TXT dos Top 5 clusters para a métrica informada.
    """
    os.makedirs(pasta_saida, exist_ok=True)
    top5 = selecionar_top_clusters(X, labels, metrica=metrica)

    colunas_desejadas = [
        "IDENTIFIER",
        "Gene symbol",
        "Gene title",
        "GO:Process",
        "GO:Function",
        "GO:Component",
    ]
    colunas_validas = [c for c in colunas_desejadas if c in df_limpo.columns]

    print("\n" + "=" * 65)
    print(f"VALIDAÇÃO BIOLÓGICA — TOP 5 CLUSTERS ({metrica.upper()})")
    print("=" * 65)

    for rank, (cluster_id, dist_media, qtd_genes) in enumerate(top5, 1):
        df_cluster = df_limpo[labels == cluster_id][colunas_validas].copy()

        #salva CSV completo do cluster
        csv_path = os.path.join(pasta_saida, f"top{rank}_cluster_{cluster_id}.csv")
        df_cluster.to_csv(csv_path, index=False)

        #salva TXT com lista de genes
        genes_lista = df_cluster["IDENTIFIER"].dropna().astype(str).tolist()
        txt_path = os.path.join(pasta_saida, f"top{rank}_cluster_{cluster_id}_genes.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(genes_lista))

        print(f"\n#{rank} | CLUSTER {cluster_id} ({qtd_genes} genes | Dist. média: {dist_media:.4f})")
        print("-" * 65)

        if "GO:Process" in df_cluster.columns:
            top_go = extrair_top_go(df_cluster["GO:Process"], top_n=5)
            print("Principais processos biológicos (GO:Process):")
            for termo, freq in top_go:
                print(f"  - {termo} ({freq} genes)")

        print("\nAmostra dos 5 primeiros genes:")
        for _, row in df_cluster.head(5).iterrows():
            simbolo = row.get("Gene symbol", row.get("IDENTIFIER", ""))
            titulo = row.get("Gene title", "")
            print(f"  • {simbolo}: {titulo}")

    print(f"\n-> Arquivos salvos em: {pasta_saida}")


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    caminho_gds = os.path.join(script_dir, "GDS38_full.soft.gz")
    caminho_npz = os.path.join(script_dir, "resultados_k100.npz")

    df_bruto = carregar_gds(caminho_gds)
    X, df_limpo, _ = preparar_dados(df_bruto)
    dados = np.load(caminho_npz)

    validar_metrica(
        X, df_limpo, dados["labels_euc"], "euclidiana",
        os.path.join(script_dir, "clusters_euclidiana"),
    )
    validar_metrica(
        X, df_limpo, dados["labels_pea"], "pearson",
        os.path.join(script_dir, "clusters_pearson"),
    )