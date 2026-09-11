# tropadoml

Trabalho final da disciplina de Machine Learning, MBA em Ciência de Dados e IA.

**Professor:** André Juan Costa Vieira
**Tema:** Qualidade de vinhos (Vini Tradizionali di Manduria)

## Equipe

| Bloco | Responsável |
|---|---|
| Tratamento de Dados | Paulo |
| Regressão (Alcohol) | Carlos |
| Análise Exploratória | Bia |
| Modelos Supervisionados + Desafio Final | Julia |
| Apresentação final | Carlos |

## Estrutura

```
tropadoml/
├── RELATORIO.md               # relatório final (método e resultados de cada bloco)
├── site/                      # apresentação interativa (Reveal.js + Plotly), publicada no GitHub Pages
│   ├── index.html
│   ├── build_data.py          # reexecuta o notebook e gera site/data/data.js
│   └── data/data.js
├── notebooks/
│   └── Trabalho Final - Machine Learning - Classificação.ipynb
└── data/
    ├── wines.csv                  # dataset validado/completo
    ├── wines_pre_processing.csv   # dataset com valores faltantes/incongruentes
    ├── wine_classification.csv    # dataset para modelos de classificação
    └── desafio.csv                # dataset do desafio final
```

## Apresentação

Slides interativos em `site/`, publicados pelo GitHub Pages em https://carloshenriquebz.github.io/tropadoml/ a cada push na `main` (workflow em `.github/workflows/pages.yml`).

- Abrir localmente: `python -m http.server 8765` dentro de `site/` e acessar http://localhost:8765 (ou abrir `site/index.html` direto no navegador).
- Regenerar os dados dos gráficos depois de mudar o notebook: `python site/build_data.py` (leva cerca de um minuto; precisa de xgboost, catboost e scikit-learn).
- Na apresentação: setas para navegar, `S` abre as notas do apresentador, `F` tela cheia, `?` lista os atalhos.

## Fluxo de trabalho

Commits direto na `main`. Sem PR (trabalho simples, time pequeno).

Convenção de commit:
```
feat(tratamento): trata valores faltantes e incongruentes
feat(regressao): implementa logistica e polinomial p/ Alcohol
feat(eda): adiciona graficos countplot/jointplot/correlacao
feat(modelos): pipeline classificacao + desafio final
```

Antes de commitar, limpar output das células do notebook (evita diffs gigantes).
