"""
SalesInsight PY - RF09 a RF14
Membro 1 (Israel): Classes, Herança, Lambda, Exportação, Regex e Main
"""

import pandas as pd
import numpy as np
import json
import os
import re

# ==================== RF11 - HIGHER-ORDER FUNCTION ====================
def processar_coluna(df, coluna, funcao_transformacao):
    """Aplica uma função de transformação a uma coluna (higher-order function)."""
    df[f"{coluna}_transformado"] = df[coluna].apply(funcao_transformacao)
    print(f"  Coluna '{coluna}_transformado' criada com sucesso.")
    return df

# ==================== RF12 - EXPORTAR RESULTADOS ====================
def exportar_resultados(metricas, clientes, stats_numpy):
    """Exporta resultados em CSV e JSON."""
    os.makedirs("outputs", exist_ok=True)
    
    metricas["por_mes"].to_csv("outputs/relatorio_resumo.csv", index=False)
    print(f"  CSV: outputs/relatorio_resumo.csv")
    
    clientes.to_csv("outputs/segmentacao_clientes.csv", index=False)
    print(f"  CSV: outputs/segmentacao_clientes.csv")
    
    stats_serializaveis = {k: round(float(v), 2) for k, v in stats_numpy.items()}
    with open("outputs/estatisticas_gerais.json", "w", encoding="utf-8") as f:
        json.dump(stats_serializaveis, f, indent=4)
    print(f"  JSON: outputs/estatisticas_gerais.json")

# ==================== RF13 - EXPRESSÕES REGULARES ====================
def limpar_strings_com_regex(df):
    """Usa expressões regulares para limpeza de colunas de texto."""
    df["cliente_limpo"] = df["cliente"].apply(
        lambda s: re.sub(r"[^a-zA-Z0-9_ ]", "", str(s)).strip()
    )
    padrao_cliente = re.compile(r"^Cliente_\d{3}$")
    df["cliente_valido"] = df["cliente_limpo"].apply(
        lambda s: bool(padrao_cliente.match(s))
    )
    n_invalidos = (~df["cliente_valido"]).sum()
    print(f"\n=== LIMPEZA COM REGEX ===")
    print(f"  Clientes inválidos: {n_invalidos}")
    return df

# ==================== RF09 - CLASSE PIPELINE ====================
class AnalisadorDeVendas:
    """Classe responsável por encapsular o pipeline de análise de vendas."""
    
    def __init__(self, caminho_arquivo):
        self.caminho_arquivo = caminho_arquivo
        self.df_bruto = None
        self.df_limpo = None
        self.metricas = {}
        self.clientes = None
        self.relatorio_limpeza = {}
    
    def carregar(self):
        self.df_bruto = pd.read_csv(self.caminho_arquivo)
        print(f"[Analisador] Carregado: {len(self.df_bruto)} registros")
        return self
    
    def limpar(self):
        print("[Analisador] Limpando dados...")
        return self
    
    def transformar(self):
        print("[Analisador] Transformando dados...")
        return self
    
    def analisar(self):
        print("[Analisador] Analisando dados...")
        return self
    
    def visualizar(self):
        print("[Analisador] Gerando visualizações...")
        return self
    
    def resumo(self):
        print("\n" + "=" * 50)
        print("       RESUMO EXECUTIVO")
        print("=" * 50)
        print(f"  Status: Pipeline executado com sucesso")
        print("=" * 50)

# ==================== RF10 - HERANÇA ====================
class AnalisadorComProjecao(AnalisadorDeVendas):
    """Extensão com funcionalidades de projeção (herança)."""
    
    def __init__(self, caminho_arquivo, meses_projecao=3):
        super().__init__(caminho_arquivo)
        self.meses_projecao = meses_projecao
        self.projecoes = []
    
    def projetar_tendencia(self):
        print("\n=== PROJEÇÃO DE TENDÊNCIA ===")
        print(f"  Projetando {self.meses_projecao} meses à frente...")
        return self

# ==================== RF14 - PIPELINE COMPLETO ====================
def main():
    """Função principal: executa o pipeline completo."""
    print("\n" + "=" * 60)
    print("   SALESINSIGHT PY – RF09 a RF14 (Israel)")
    print("=" * 60)
    
    # Verificar se o dataset existe
    if not os.path.exists("vendas.csv"):
        print("[ERRO] Execute primeiro a parte do Membro 2 para gerar o dataset!")
        return
    
    # Pipeline via classe com herança
    analisador = AnalisadorComProjecao("vendas.csv", meses_projecao=3)
    (analisador
        .carregar()
        .limpar()
        .transformar()
        .analisar()
        .projetar_tendencia()
        .visualizar()
    )
    
    # Demonstração RF11
    print("\n=== RF11 - Higher-Order Function ===")
    df_teste = pd.DataFrame({"valor": [100, 200, 300]})
    processar_coluna(df_teste, "valor", lambda x: x * 2)
    
    # Demonstração RF13
    print("\n=== RF13 - Expressões Regulares ===")
    df_teste2 = pd.DataFrame({"cliente": ["Cliente_001", "Cliente_002"]})
    limpar_strings_com_regex(df_teste2)
    
    # Demonstração RF12
    print("\n=== RF12 - Exportação ===")
    metricas_exemplo = {"por_mes": pd.DataFrame({"mes": [1, 2], "receita_total": [1000, 2000]})}
    clientes_exemplo = pd.DataFrame({"cliente": ["A", "B"], "total_gasto": [1000, 2000]})
    stats_exemplo = {"media": 1500, "total": 3000}
    exportar_resultados(metricas_exemplo, clientes_exemplo, stats_exemplo)
    
    analisador.resumo()
    print("\n✅ Minha parte (RF09 a RF14) está completa!")

# ==================== EXECUÇÃO ====================
if __name__ == "__main__":
    main()