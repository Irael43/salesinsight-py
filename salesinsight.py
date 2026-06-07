"""
SalesInsight PY - Pipeline de Análise e Visualização de Dados de Vendas
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import json
import os
from datetime import datetime, timedelta
import random

# Configurações
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ==================== RF01 - GERAR DATASET ====================
def gerar_dataset_vendas(n_registros=200, seed=42):
    random.seed(seed)
    np.random.seed(seed)
    
    produtos = ["Notebook", "Smartphone", "Tablet", "Monitor", "Teclado", "Mouse", "Headset"]
    categorias = {"Notebook": "Computadores", "Smartphone": "Celulares", "Tablet": "Celulares",
                  "Monitor": "Computadores", "Teclado": "Periféricos", "Mouse": "Periféricos",
                  "Headset": "Periféricos"}
    regioes = ["Sudeste", "Sul", "Nordeste", "Centro-Oeste", "Norte"]
    clientes = [f"Cliente_{i:03d}" for i in range(1, 51)]
    data_inicio = datetime(2024, 1, 1)
    dados = []
    
    for i in range(n_registros):
        produto = random.choice(produtos)
        quantidade = random.randint(1, 10)
        preco_base = {"Notebook": 3500, "Smartphone": 2200, "Tablet": 1800,
                      "Monitor": 1200, "Teclado": 250, "Mouse": 120, "Headset": 350}[produto]
        preco = round(preco_base * random.uniform(0.85, 1.15), 2)
        data = data_inicio + timedelta(days=random.randint(0, 364))
        
        if random.random() < 0.05:
            quantidade = None
        if random.random() < 0.04:
            preco = None
        if random.random() < 0.03:
            produto = "  " + produto
            
        dados.append({
            "id_venda": i + 1,
            "data_venda": data.strftime("%Y-%m-%d") if random.random() > 0.02 else "DATA INVÁLIDA",
            "cliente": random.choice(clientes),
            "produto": produto,
            "categoria": categorias.get(produto.strip(), "Outros"),
            "regiao": random.choice(regioes),
            "quantidade": quantidade,
            "preco_unitario": preco
        })
    
    return pd.DataFrame(dados)

# ==================== RF02 - INSPECIONAR ====================
def inspecionar_dados(df):
    print("\n=== INSPEÇÃO INICIAL DO DATASET ===")
    print(f"Shape: {df.shape}")
    print(f"\nColunas: {list(df.columns)}")
    print(f"\nTipos de dados:\n{df.dtypes}")
    print(f"\nValores nulos por coluna:\n{df.isnull().sum()}")
    print(f"\nPrimeiros registros:\n{df.head()}")
    print(f"\nEstatísticas descritivas:\n{df.describe()}")

# ==================== RF03 - LIMPAR DADOS ====================
def limpar_dados(df):
    n_inicial = len(df)
    relatorio = {}
    
    colunas_texto = df.select_dtypes(include="object").columns
    for col in colunas_texto:
        df[col] = df[col].str.strip()
    
    df["data_venda"] = pd.to_datetime(df["data_venda"], errors="coerce")
    n_datas_invalidas = df["data_venda"].isnull().sum()
    df = df.dropna(subset=["data_venda"])
    relatorio["datas_invalidas_removidas"] = n_datas_invalidas
    
    n_antes = len(df)
    df = df.dropna(subset=["quantidade", "preco_unitario"])
    relatorio["linhas_nulas_removidas"] = n_antes - len(df)
    
    df["quantidade"] = df["quantidade"].astype(int)
    df["preco_unitario"] = df["preco_unitario"].astype(float)
    
    relatorio["registros_iniciais"] = n_inicial
    relatorio["registros_finais"] = len(df)
    relatorio["registros_removidos_total"] = n_inicial - len(df)
    
    print("\n=== RELATÓRIO DE LIMPEZA ===")
    for chave, valor in relatorio.items():
        print(f"  {chave}: {valor}")
    
    return df, relatorio

# ==================== RF04 - COLUNAS DERIVADAS ====================
def criar_colunas_derivadas(df):
    df["receita_total"] = df["quantidade"] * df["preco_unitario"]
    df["mes"] = df["data_venda"].dt.month
    df["mes_nome"] = df["data_venda"].dt.strftime("%B")
    df["trimestre"] = df["data_venda"].dt.quarter.apply(lambda q: f"Q{q}")
    df["ano"] = df["data_venda"].dt.year
    
    condicoes = [
        df["receita_total"] < 500,
        (df["receita_total"] >= 500) & (df["receita_total"] < 5000),
        df["receita_total"] >= 5000
    ]
    classificacoes = ["Baixo Valor", "Médio Valor", "Alto Valor"]
    df["faixa_receita_item"] = np.select(condicoes, classificacoes, default="Não Classificado")
    
    print("\n=== COLUNAS DERIVADAS CRIADAS ===")
    print(df[["data_venda", "receita_total", "mes", "trimestre", "faixa_receita_item"]].head())
    
    return df

# ==================== RF05 - MÉTRICAS AGREGADAS ====================
def calcular_metricas(df):
    metricas = {}
    
    por_mes = df.groupby("mes").agg(
        receita_total=("receita_total", "sum"),
        quantidade=("quantidade", "sum"),
        n_vendas=("id_venda", "count")
    ).reset_index().sort_values("mes")
    metricas["por_mes"] = por_mes
    
    top_produtos = df.groupby("produto")["receita_total"].sum().sort_values(ascending=False).head(5).reset_index()
    metricas["top_produtos"] = top_produtos
    
    por_categoria = df.groupby("categoria")["receita_total"].sum().reset_index()
    metricas["por_categoria"] = por_categoria
    
    por_regiao = df.groupby("regiao").agg(
        receita_total=("receita_total", "sum"),
        media_ticket=("receita_total", "mean")
    ).reset_index().sort_values("receita_total", ascending=False)
    metricas["por_regiao"] = por_regiao
    
    for nome, tabela in metricas.items():
        print(f"\n=== {nome.upper().replace('_', ' ')} ===")
        print(tabela.to_string(index=False))
    
    return metricas

# ==================== RF06 - SEGMENTAR CLIENTES ====================
def segmentar_clientes(df):
    clientes = df.groupby("cliente")["receita_total"].sum().reset_index()
    clientes.columns = ["cliente", "total_gasto"]
    
    clientes["segmento"] = clientes["total_gasto"].apply(
        lambda gasto: "Ouro" if gasto > 15000 else ("Prata" if gasto >= 5000 else "Bronze")
    )
    
    clientes = clientes.sort_values("total_gasto", ascending=False)
    
    print("\n=== SEGMENTAÇÃO DE CLIENTES ===")
    print(clientes.head(10).to_string(index=False))
    print(f"\nDistribuição de segmentos:\n{clientes['segmento'].value_counts()}")
    
    return clientes

# ==================== RF07 - ESTATÍSTICAS NUMPY ====================
def calcular_estatisticas_numpy(df):
    print("\n=== ESTATÍSTICAS COM NUMPY ===")
    receitas = df["receita_total"].to_numpy()
    
    media = np.mean(receitas)
    mediana = np.median(receitas)
    desvio_padrao = np.std(receitas)
    total = np.sum(receitas)
    p25 = np.percentile(receitas, 25)
    p75 = np.percentile(receitas, 75)
    
    print(f"  Receita média por venda:    R$ {media:.2f}")
    print(f"  Receita mediana por venda:  R$ {mediana:.2f}")
    print(f"  Desvio padrão:              R$ {desvio_padrao:.2f}")
    print(f"  Receita total:              R$ {total:.2f}")
    print(f"  Percentil 25 (Q1):          R$ {p25:.2f}")
    print(f"  Percentil 75 (Q3):          R$ {p75:.2f}")
    
    receitas_normalizadas = (receitas - receitas.min()) / (receitas.max() - receitas.min())
    print(f"\n  Receitas normalizadas (primeiros 5): {receitas_normalizadas[:5].round(4)}")
    
    acima_da_media = receitas[receitas > media]
    print(f"\n  Vendas acima da média: {len(acima_da_media)} de {len(receitas)}")
    
    return {"media": media, "mediana": mediana, "desvio_padrao": desvio_padrao, "total": total}

# ==================== RF08 - VISUALIZAÇÕES ====================
def gerar_visualizacoes(df, metricas, output_dir="outputs/graficos"):
    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams["figure.figsize"] = (12, 6)
    
    # Gráfico 1: Receita por Mês (linha)
    fig, ax = plt.subplots()
    por_mes = metricas["por_mes"]
    ax.plot(por_mes["mes"], por_mes["receita_total"], marker="o", linewidth=2, color="#2196F3")
    ax.fill_between(por_mes["mes"], por_mes["receita_total"], alpha=0.15, color="#2196F3")
    ax.set_title("Receita Total por Mês (2024)", fontsize=14)
    ax.set_xlabel("Mês", fontsize=12)
    ax.set_ylabel("Receita Total (R$)", fontsize=12)
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(["Jan", "Fev", "Mar", "Abr", "Mai", "Jun", "Jul", "Ago", "Set", "Out", "Nov", "Dez"], rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "vendas_por_mes.png"), dpi=150)
    plt.close()
    print(f"  Gráfico exportado: {output_dir}/vendas_por_mes.png")
    
    # Gráfico 2: Top 5 Produtos (barras)
    fig, ax = plt.subplots()
    top = metricas["top_produtos"]
    sns.barplot(data=top, y="produto", x="receita_total", ax=ax, palette="Blues_d")
    ax.set_title("Top 5 Produtos por Receita Total", fontsize=14)
    ax.set_xlabel("Receita Total (R$)", fontsize=12)
    ax.set_ylabel("Produto", fontsize=12)
    for container in ax.containers:
        ax.bar_label(container, fmt="R$ %.0f", padding=5)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "top_produtos.png"), dpi=150)
    plt.close()
    print(f"  Gráfico exportado: {output_dir}/top_produtos.png")
    
    # Gráfico 3: Boxplot por Região
    fig, ax = plt.subplots()
    sns.boxplot(data=df, x="regiao", y="receita_total", ax=ax, palette="Set2")
    ax.set_title("Distribuição de Receita por Transação – Por Região", fontsize=14)
    ax.set_xlabel("Região", fontsize=12)
    ax.set_ylabel("Receita por Venda (R$)", fontsize=12)
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "distribuicao_regioes.png"), dpi=150)
    plt.close()
    print(f"  Gráfico exportado: {output_dir}/distribuicao_regioes.png")
    
    print("\n=== VISUALIZAÇÕES GERADAS COM SUCESSO ===")

# ==================== RF09 e RF10 - CLASSES ====================
class AnalisadorDeVendas:
    def __init__(self, caminho_arquivo):
        self.caminho_arquivo = caminho_arquivo
        self.df_bruto = None
        self.df_limpo = None
        self.metricas = {}
        self.clientes = None
        self.relatorio_limpeza = {}
    
    def carregar(self):
        self.df_bruto = pd.read_csv(self.caminho_arquivo)
        print(f"[AnalisadorDeVendas] Arquivo carregado: {self.caminho_arquivo}")
        print(f"  Registros carregados: {len(self.df_bruto)}")
        return self
    
    def limpar(self):
        self.df_limpo, self.relatorio_limpeza = limpar_dados(self.df_bruto.copy())
        return self
    
    def transformar(self):
        self.df_limpo = criar_colunas_derivadas(self.df_limpo)
        return self
    
    def analisar(self):
        self.metricas = calcular_metricas(self.df_limpo)
        self.clientes = segmentar_clientes(self.df_limpo)
        return self
    
    def visualizar(self):
        gerar_visualizacoes(self.df_limpo, self.metricas)
        return self
    
    def resumo(self):
        print("\n" + "=" * 50)
        print("       RESUMO EXECUTIVO – SALESINSIGHT PY")
        print("=" * 50)
        print(f"  Registros brutos:       {self.relatorio_limpeza.get('registros_iniciais', 'N/A')}")
        print(f"  Registros limpos:       {self.relatorio_limpeza.get('registros_finais', 'N/A')}")
        receita = self.df_limpo["receita_total"].sum() if self.df_limpo is not None else 0
        print(f"  Receita total anual:    R$ {receita:,.2f}")
        if self.clientes is not None:
            top = self.clientes.iloc[0]
            print(f"  Cliente top:            {top['cliente']} (R$ {top['total_gasto']:,.2f})")
        print("=" * 50)

class AnalisadorComProjecao(AnalisadorDeVendas):
    def __init__(self, caminho_arquivo, meses_projecao=3):
        super().__init__(caminho_arquivo)
        self.meses_projecao = meses_projecao
        self.projecoes = []
    
    def projetar_tendencia(self):
        if not self.metricas or "por_mes" not in self.metricas:
            print("[AVISO] Rode .analisar() antes de projetar.")
            return self
        
        por_mes = self.metricas["por_mes"].sort_values("mes")
        receitas_historicas = por_mes["receita_total"].to_numpy()
        ultimos_3 = receitas_historicas[-3:]
        media_movel = np.mean(ultimos_3)
        tendencia = np.std(ultimos_3) * 0.1
        ultimo_mes = int(por_mes["mes"].max())
        
        print("\n=== PROJEÇÃO DE TENDÊNCIA (Média Móvel Simples) ===")
        print(f"  Base: média dos últimos 3 meses = R$ {media_movel:,.2f}")
        self.projecoes = []
        
        for i in range(1, self.meses_projecao + 1):
            mes_projetado = (ultimo_mes + i - 1) % 12 + 1
            receita_projetada = media_movel + (tendencia * i)
            self.projecoes.append({"mes": mes_projetado, "receita_projetada": round(receita_projetada, 2)})
            print(f"  Mês {mes_projetado:02d} (projeção): R$ {receita_projetada:,.2f}")
        return self

# ==================== RF11 - HIGHER-ORDER FUNCTION ====================
def processar_coluna(df, coluna, funcao_transformacao):
    df[f"{coluna}_transformado"] = df[coluna].apply(funcao_transformacao)
    print(f"  Coluna '{coluna}_transformado' criada com sucesso.")
    return df

# ==================== RF13 - EXPRESSÕES REGULARES ====================
def limpar_strings_com_regex(df):
    df["cliente_limpo"] = df["cliente"].apply(
        lambda s: re.sub(r"[^a-zA-Z0-9_ ]", "", str(s)).strip()
    )
    padrao_cliente = re.compile(r"^Cliente_\d{3}$")
    df["cliente_valido"] = df["cliente_limpo"].apply(
        lambda s: bool(padrao_cliente.match(s))
    )
    n_invalidos = (~df["cliente_valido"]).sum()
    print(f"\n=== LIMPEZA COM REGEX ===")
    print(f"  Clientes com formato inválido: {n_invalidos}")
    return df

# ==================== RF12 - EXPORTAR RESULTADOS ====================
def exportar_resultados(metricas, clientes, stats_numpy):
    os.makedirs("outputs", exist_ok=True)
    metricas["por_mes"].to_csv("outputs/relatorio_resumo.csv", index=False, encoding="utf-8-sig")
    print(f"  CSV exportado: outputs/relatorio_resumo.csv")
    clientes.to_csv("outputs/segmentacao_clientes.csv", index=False, encoding="utf-8-sig")
    print(f"  CSV exportado: outputs/segmentacao_clientes.csv")
    stats_serializaveis = {k: round(float(v), 2) for k, v in stats_numpy.items()}
    with open("outputs/estatisticas_gerais.json", "w", encoding="utf-8") as f:
        json.dump(stats_serializaveis, f, indent=4, ensure_ascii=False)
    print(f"  JSON exportado: outputs/estatisticas_gerais.json")

# ==================== RF14 - MAIN ====================
def main():
    print("\n" + "=" * 60)
    print("   SALESINSIGHT PY – Pipeline de Análise de Dados de Vendas")
    print("=" * 60)
    
    if not os.path.exists("vendas.csv"):
        print("\n[INFO] Gerando dataset sintético...")
        df_gerado = gerar_dataset_vendas(n_registros=200)
        df_gerado.to_csv("vendas.csv", index=False)
    
    analisador = AnalisadorComProjecao("vendas.csv", meses_projecao=3)
    (analisador
        .carregar()
        .limpar()
        .transformar()
        .analisar()
        .projetar_tendencia()
        .visualizar()
    )
    
    stats = calcular_estatisticas_numpy(analisador.df_limpo)
    exportar_resultados(analisador.metricas, analisador.clientes, stats)
    analisador.df_limpo = limpar_strings_com_regex(analisador.df_limpo)
    analisador.resumo()
    
    print("\n[CONCLUÍDO] Pipeline finalizado com sucesso!")

if __name__ == "__main__":
    main()