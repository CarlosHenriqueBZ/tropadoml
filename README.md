# tropadoml

Trabalho final da disciplina de Machine Learning — Pós-graduação em Inteligência Artificial para Negócios.

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
├── notebooks/
│   └── Trabalho Final - Machine Learning - Classificação.ipynb
└── data/
    ├── wines.csv                  # dataset validado/completo
    ├── wines_pre_processing.csv   # dataset com valores faltantes/incongruentes
    ├── wine_classification.csv    # dataset para modelos de classificação
    └── desafio.csv                # dataset do desafio final
```

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
