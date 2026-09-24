# Trabalho Prático 1 - Recuperação de Informação (SCC0282 - 2026.2)


## Integrante
- Gabriel Hyppolito - 14571810

## Descrição
Sistema de recuperação textual sobre a coleção Cranfield, implementando o
Modelo Vetorial (TF-IDF + similaridade do cosseno) e o Modelo Probabilístico
BM25, com comparação quantitativa via Precision@10, Recall@10 e MAP.

## Base de dados
- **Cranfield**, obtida via biblioteca [`ir_datasets`](https://ir-datasets.com/cranfield.html)
  (documentos, consultas e julgamentos de relevância/qrels).
- Download automático na primeira execução de `src/data_loader.py`.

## Requisitos e instalação
- Python 3.10+
- Instalar dependências:
  ```bash
  pip install -r requirements.txt
  ```
- Baixar recursos do NLTK (feito automaticamente por `src/preprocessing.py`,
  mas pode ser feito manualmente):
  ```python
  import nltk
  nltk.download('punkt')
  nltk.download('punkt_tab')
  nltk.download('stopwords')
  ```

## Principais bibliotecas utilizadas
- `ir_datasets` — acesso à coleção Cranfield
- `nltk` — tokenização, stopwords, stemming (Porter)
- `scikit-learn` — TF-IDF (`TfidfVectorizer`) e similaridade do cosseno
- `pandas` / `numpy` — manipulação de dados

## Como executar
Execute os scripts na ordem abaixo, a partir da raiz do repositório:

```bash
# 1. Baixa e materializa os dados brutos em dados/raw/
python -m src.data_loader

# 2. Gera as 4 configurações de pré-processamento em dados/processed/
python -m src.preprocessing

# 3. Roda os testes (um script por requisito do TP)
python -m testes.compara_preprocessamento     # requisito 1
python -m testes.compara_modelos              # requisitos 2, 3, 4, 5
python -m testes.analise_consultas            # requisito 6
python -m testes.testa_param_bm25             # requisito 7
python -m testes.altera_consultas             # requisito 8
python -m testes.analise_erros                # requisito 9
```

Além dos scripts acima, `src/consulta_query_doc.py` é um utilitário de
inspeção pontual (não faz parte da sequência principal): edite as
constantes `QUERY_ID`/`DOC_ID` no topo do arquivo e rode
`python -m src.consulta_query_doc` para ver o texto de uma consulta e de um
documento específicos lado a lado, com diagnósticos de TF/DF/IDF, útil
para investigar hipóteses ao escrever a análise crítica do relatório.

Os resultados são salvos em:
- `resultados/por_query/` — métricas por consulta, por modelo/configuração
- `resultados/agregados/` — métricas agregadas, tabelas de diferença/erro e
  a grade de parâmetros do BM25 (uma CSV por requisito)

## Configuração fixada nos experimentos (requisitos 5-9)
A configuração de pré-processamento usada como padrão a partir do
requisito 5 é `ambas` (remoção de stopwords + stemming), escolhida por ter
obtido o maior MAP no requisito 1 (ver
`resultados/agregados/compara_preprocessamento.csv`). O BM25 usa
`k1=1.2, b=0.75` como parâmetros padrão fora do requisito 7 (que varia
justamente esses valores). Essas escolhas estão centralizadas na constante
`CONFIG_NAME` no topo de cada script em `testes/`.

## Estrutura do repositório
```
tp1-ri/
├── dados/raw/          # docs, queries, qrels materializados do ir_datasets
├── dados/processed/    # tokens pré-processados (4 configurações)
├── src/                # módulos principais (pré-processamento, modelos, métricas)
├── testes/             # um script por requisito do enunciado
├── resultados/         # saídas: CSVs por consulta e agregados
└── report/             # relatório em PDF
```
