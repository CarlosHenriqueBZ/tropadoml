# Relatório do trabalho final: qualidade de vinhos

Disciplina de Machine Learning, pós-graduação em Inteligência Artificial para Negócios. Professor André Juan Costa Vieira.

| Bloco | Responsável |
|---|---|
| Tratamento de dados | Paulo |
| Regressão para preencher Alcohol | Carlos |
| Análise exploratória | Bia |
| Modelos supervisionados e desafio final | Julia |
| Relatório e apresentação | Carlos |

O código completo está no notebook `notebooks/Trabalho Final - Machine Learning - Classificação.ipynb`. Este relatório resume o método e os resultados de cada etapa. Todos os números abaixo saíram de uma reexecução do notebook em 11/09/2026.

## 1. Objetivo

A vinícola entregou uma planilha com medidas físico-químicas de 6.497 vinhos: onze variáveis contínuas (acidez, açúcar residual, cloretos, dióxido de enxofre, densidade, pH, sulfatos e teor alcoólico), a cor (tinto ou branco) e a nota de qualidade, de 3 a 9. O trabalho tem quatro etapas. Primeiro, corrigir a planilha, que chegou com células vazias e valores adulterados. Depois, preencher o teor alcoólico que faltava usando modelos de regressão, em vez da mediana. Em seguida, explorar os dados com gráficos e correlações. Por fim, treinar classificadores de qualidade e aplicar o resultado a uma base de garrafas novas, para escolher quais servir a um crítico.

## 2. Dados

| Arquivo | Linhas | Uso |
|---|---|---|
| wines_pre_processing.csv | 6.497 | planilha danificada, ponto de partida do tratamento |
| wines.csv | 6.497 | mesma planilha íntegra, usada como gabarito |
| wine_classification.csv | 13.701 | treino dos classificadores de qualidade, notas de 3 a 8 |
| desafio.csv | 13.890 | garrafas novas, com nota e sem a coluna de cor |

No wines.csv há 4.898 brancos e 1.599 tintos. As notas se concentram em 5, 6 e 7: 94,9% dos tintos e 92,6% dos brancos estão nessa faixa. Só 30 vinhos têm nota 3 e só 5 têm nota 9, todos brancos.

## 3. Tratamento dos dados

A planilha danificada tinha dois tipos de problema. O primeiro eram 41 células vazias espalhadas por doze colunas, no máximo 7 por coluna. O segundo eram 137 células com texto onde deveria haver número ou cor: "starwars", "deep learning", "my precious", "vinho ruim", "bart", entre outros. Foram 11 textos em cada uma das onze colunas físico-químicas, 11 na coluna de cor e 5 na de qualidade, onde o valor adulterado era o índice da linha entre colchetes.

O tratamento converteu cada coluna numérica com `pd.to_numeric(errors='coerce')`, o que transforma texto em vazio e permite tratar os dois problemas de uma vez. Nas dez colunas contínuas o vazio recebeu a mediana da coluna. Em qualidade e cor, que são discretas, entrou a moda. A coluna de teor alcoólico foi deixada de propósito com seus 15 vazios (4 originais e 11 textos), porque ela é o alvo da etapa de regressão.

A validação comparou o resultado com o wines.csv. A média de cada coluna mudou menos de 0,03% e o desvio padrão só mudou na terceira casa decimal, então a distribuição foi preservada. Os erros ficaram nas colunas discretas. Das 14 linhas sem cor válida (3 vazias e 11 com texto), 3 eram tintos e receberam a moda, branco. Das 5 linhas sem nota, 3 receberam a moda 6 quando a nota verdadeira era 4 ou 7. A mediana e a moda funcionam bem quando o volume de faltantes é pequeno, como aqui, e quando o valor a preencher não precisa ser exato. O teor alcoólico foi tratado à parte justamente por ser o caso em que a mediana erra mais, como mostra a seção seguinte.

## 4. Preenchimento do teor alcoólico por regressão

### 4.1 Regressão logística

O enunciado pede uma regressão logística para preencher o teor alcoólico. A logística é um algoritmo de classificação: a saída é a probabilidade de cada categoria. O teor alcoólico é um número contínuo, vai de 8,0 a 14,9 e tem mais de cem valores distintos na planilha. Para usar a logística foi preciso transformar o alvo em categorias. O teor foi cortado em três faixas por tercil, cada uma com cerca de um terço dos vinhos: baixo (até 9,7), médio (até 11,0) e alto (acima de 11,0). O modelo prevê a faixa e a linha recebe a mediana da faixa prevista: 9,3, 10,4 ou 11,9.

As outras onze colunas entraram como variáveis, padronizadas com StandardScaler, e a divisão foi 80% treino e 20% teste, estratificada por faixa. No teste, com 1.297 vinhos:

| Métrica | Valor |
|---|---|
| Acurácia | 0,824 |
| Precisão (macro) | 0,828 |
| Recall (macro) | 0,827 |
| F1 (macro) | 0,828 |
| ROC AUC (um contra o resto) | 0,945 |

A faixa do meio é a mais confundida (F1 de 0,75, contra 0,84 e 0,88 nas outras), o que é esperado, pois ela faz fronteira com as duas vizinhas. Nas 15 linhas que faltavam, comparando com o wines.csv, o modelo acertou a faixa de 13 e teve AUC de 0,980. Mesmo assim o erro em graus de álcool foi de 0,51 em média (MAE), com RMSE de 0,686. Um vinho de 13,1 graus, por exemplo, caiu na faixa certa e recebeu 11,9. O corte em faixas descarta a informação de onde o vinho está dentro da faixa, e ela não volta na hora de preencher.

### 4.2 Regressão polinomial

Na regressão polinomial o alvo volta a ser o número. O pipeline padroniza as colunas, gera os termos polinomiais (quadrados, cubos e produtos entre colunas) e ajusta uma regressão linear. Os graus 1, 2 e 3 foram comparados por validação cruzada de cinco partes no treino, antes de olhar o teste:

| Grau | Termos gerados | R² médio na validação cruzada |
|---|---|---|
| 1 | 12 | 0,803 (desvio 0,069) |
| 2 | 90 | 0,867 (desvio 0,018) |
| 3 | 454 | 0,135 (desvio 1,362) |

O grau 3 cria 454 colunas a partir de 12 e decora o treino: em algumas partes da validação o R² fica negativo. O grau 2 foi o escolhido. No teste:

| Métrica | Valor |
|---|---|
| MAE | 0,312 |
| MSE | 0,207 |
| RMSE | 0,455 |
| MAPE | 2,99% |
| R² | 0,854 |

O pior erro do teste é um tinto com cloretos de 0,61 e sulfatos de 2,0, os maiores valores do dataset inteiro. O polinômio eleva esses valores ao quadrado e a previsão cai para 3,2 graus quando o real é 9,4. É o custo conhecido dos modelos polinomiais fora da região onde foram treinados.

Nas 15 linhas faltantes, quatro técnicas foram comparadas contra o wines.csv. O KNNImputer entrou como contraprova de um imputador pronto: cada linha recebe a média do teor alcoólico dos 5 vinhos mais parecidos, ponderada pela distância.

| Técnica | MAE | RMSE | R² |
|---|---|---|---|
| Mediana da coluna | 1,093 | 1,351 | -0,041 |
| Logística em 3 faixas | 0,513 | 0,686 | 0,732 |
| Polinomial grau 2 | 0,399 | 0,550 | 0,828 |
| KNNImputer (k = 5) | 0,387 | 0,594 | 0,798 |

A polinomial reduz o erro da mediana em 63% e o da logística em 22%. O KNN empata no MAE e perde no RMSE, que pesa mais os erros grandes. Com 15 amostras a diferença entre os dois é ruído, e o empate não justifica trocar a técnica pedida. A planilha final recebeu os valores da polinomial de grau 2. A variável que mais ajuda o modelo é a densidade, com correlação de -0,69 com o teor alcoólico: durante a fermentação o açúcar, mais denso que a água, vira álcool, menos denso, então vinho com mais álcool é menos denso.

O e-mail pedido na questão 5, com a justificativa técnica para não usar a logística, está no notebook logo após o preenchimento.

## 5. Análise exploratória

Os gráficos foram feitos em Plotly, interativos, com as paletas Sunsetdark, YlOrBr e Tealrose e uma paleta fixa para cor do vinho (tinto em bordô, branco em dourado). Os principais achados:

A distribuição de notas é parecida entre tintos e brancos, concentrada em 5, 6 e 7. Notas extremas são raras nos dois tipos.

No gráfico de densidade de álcool contra açúcar residual, a maior concentração está em açúcar abaixo de 5 g/L (59,7% dos vinhos) e álcool entre 9% e 12,5% (88,5% dos vinhos). Há uma cauda de 18 vinhos com mais de 20 g/L de açúcar, em geral com álcool um pouco menor. O açúcar residual mediano do branco (5,2 g/L) é mais que o dobro do tinto (2,2 g/L).

O boxplot de açúcar residual por nota identificou 147 outliers pelo critério de 1,5 vezes o intervalo interquartil dentro de cada nota. Desses, 146 são brancos e apenas 1 é tinto. Eles se concentram nas notas 7 (82 vinhos) e 6 (31). O maior valor é um branco nota 6 com 65,8 g/L. Isso é coerente com a existência de brancos bem mais doces no dataset, algo raro entre os tintos.

O mapa de correlação, com o corte de 0,4 pedido no enunciado, encontrou oito pares fortes:

| Par | r |
|---|---|
| free sulfur dioxide e total sulfur dioxide | 0,72 |
| residual sugar e density | 0,55 |
| residual sugar e total sulfur dioxide | 0,50 |
| fixed acidity e density | 0,46 |
| alcohol e quality | 0,44 |
| residual sugar e free sulfur dioxide | 0,40 |
| volatile acidity e total sulfur dioxide | -0,41 |
| density e alcohol | -0,69 |

A correlação positiva mais forte, SO2 livre com SO2 total, é aritmética: o total é a soma do livre com a parte combinada a outras moléculas, então quem tem mais SO2 livre tende a ter mais total. A negativa mais forte, densidade com álcool, é a relação de fermentação explicada na seção 4. Entre as variáveis e a nota, a única correlação acima do corte é com o teor alcoólico (0,44).

## 6. Classificação da qualidade

### 6.1 Base e preparação

A base wine_classification.csv tem 13.701 linhas e notas de 3 a 8. A distribuição é bem diferente do wines.csv: a nota 3 é a mais frequente, com 4.719 linhas, e a nota 8 tem 1.271. A base também tem 6.640 linhas duplicadas, sem contar as colunas de índice. Isso indica uma base gerada ou reamostrada para balancear as classes, e traz um ponto de atenção discutido no fim desta seção.

O alvo passou por LabelEncoder para virar inteiros de 0 a 5, exigência do XGBoost. A divisão foi 80% treino e 20% teste, estratificada, com semente 12. As variáveis foram padronizadas com StandardScaler ajustado só no treino, o que importa para KNN e SVM, sensíveis a escala.

### 6.2 Pipeline com seis algoritmos

O enunciado pede ao menos cinco algoritmos, incluindo uma floresta aleatória, dois de boosting, SVM e KNN. Foram treinados seis, todos com os parâmetros padrão e 100 estimadores onde aplicável. As métricas macro dão o mesmo peso a cada nota, o que evita que as classes maiores dominem o resultado. No teste, com 2.741 vinhos:

| Modelo | Acurácia | Precisão | Recall | F1 | ROC AUC (macro) |
|---|---|---|---|---|---|
| KNN | 0,708 | 0,683 | 0,637 | 0,653 | 0,931 |
| Random Forest | 0,913 | 0,914 | 0,896 | 0,905 | 0,989 |
| CatBoost | 0,819 | 0,806 | 0,772 | 0,786 | 0,968 |
| AdaBoost | 0,603 | 0,523 | 0,507 | 0,504 | 0,855 |
| XGBoost | 0,914 | 0,909 | 0,898 | 0,903 | 0,986 |
| SVM | 0,749 | 0,733 | 0,676 | 0,695 | 0,936 |

Random Forest e XGBoost empatam na frente, com 91% de acurácia e AUC acima de 0,98. CatBoost com 100 iterações fica atrás, e AdaBoost tem o pior resultado, com dificuldade especial nas notas 7 e 8 (recall de 0,13 e 0,24). KNN e SVM ficam no meio. O gráfico de importância das variáveis, feito para Random Forest e XGBoost, mostra uma distribuição bem espalhada: o dióxido de enxofre total é a variável mais usada nos dois modelos (12% a 13% da importância) e a densidade é a menos usada (5% a 6%). As demais ficam entre 7% e 11%.

### 6.3 Otimização de hiperparâmetros

O Random Forest foi escolhido para otimização. A busca usou Hyperopt com o algoritmo TPE, 50 avaliações, e recall macro em validação cruzada estratificada de cinco partes como objetivo. O espaço cobriu número de árvores (100 a 600), profundidade máxima, mínimo de amostras por divisão e por folha, número de variáveis por divisão e critério (gini ou entropia).

A primeira busca encontrou entropia, profundidade 24, log2 variáveis por divisão, 200 árvores, mínimo de 6 amostras por divisão e 1 por folha, com recall macro de 0,832 na validação cruzada. No teste esse modelo ficou com acurácia de 0,91, recall macro de 0,89 e F1 macro de 0,90, praticamente igual ao Random Forest padrão. A limitação da profundidade pareceu ser o que segurou o ganho, então a segunda busca incluiu a opção de profundidade ilimitada e embutiu o scaler no pipeline de validação cruzada. Na reexecução, a segunda busca chegou a gini, profundidade 30, log2 variáveis por divisão, 450 árvores, mínimo de 4 amostras por divisão e 1 por folha, com recall macro de 0,834 na validação cruzada. No teste, o modelo ficou com acurácia de 0,914, F1 macro de 0,905 e AUC de 0,988, o mesmo patamar do Random Forest padrão. As duas buscas confirmam que os parâmetros padrão já extraem dessa base quase tudo que ela oferece, e que o ganho da otimização aqui está na segunda casa decimal.

### 6.4 Naive Bayes

O Naive Bayes gaussiano, pedido no fim do bloco, teve acurácia de 0,55 e F1 macro de 0,50 no mesmo teste, bem abaixo dos modelos de árvore. O motivo está na hipótese do método: ele assume que, dentro de cada nota, as variáveis são independentes entre si. Para verificar, o notebook calcula a correlação entre as variáveis separadamente para cada nota. Em todas as classes há entre 15 e 23 pares com correlação forte (acima de 0,4 em módulo), de 55 pares possíveis. A hipótese de independência não se sustenta nesses dados, e o modelo paga por isso.

### 6.5 Ponto de atenção

As 6.640 linhas duplicadas da base de classificação fazem com que parte dos vinhos do teste tenha uma cópia idêntica no treino. Isso favorece modelos que memorizam, como florestas com árvores profundas e KNN, e tende a inflar as métricas. Os 91% de acurácia devem ser lidos como desempenho nessa base, e não como expectativa para vinhos novos. Uma verificação futura seria remover as duplicatas antes da divisão e repetir o pipeline.

## 7. Desafio final

O pedido é escolher três garrafas da base desafio.csv para um crítico que só bebe tinto e só aceita vinhos de nota 8 ou 9. A base tem 13.890 garrafas, todas com nota, mas sem a coluna de cor. O problema, portanto, é descobrir a cor, e com poucos falsos positivos: servir um branco seria pior que deixar de servir um tinto.

A equipe treinou um classificador de cor no wines.csv, com as mesmas onze variáveis e o mesmo conjunto de seis algoritmos. Todos passaram de 0,99 de F1 macro no teste. Para checar sobreajuste, a validação cruzada de cinco partes comparou treino e validação: a diferença de F1 ficou abaixo de 0,008 em todos os modelos. O XGBoost foi o escolhido, com acurácia de 0,996 na validação e precisão de 0,996 para a classe tinto.

Aplicado ao desafio, o modelo previu 2.149 tintos e 11.741 brancos. Entre as 13.890 garrafas, só 11 têm nota 8 ou 9 (6 e 5, respectivamente). Dessas 11, o modelo classificou 9 como brancas e 2 como tintas, ambas de nota 8, com probabilidade de tinto de 0,78 e 0,63. Não há três tintos que atendam à regra do crítico. As duas garrafas estão registradas na última célula do notebook, e a recomendação é servir essas duas e levar à direção a decisão sobre a terceira: liberar um tinto de nota 7 (há 41 previstos, 31 deles com probabilidade de tinto acima de 0,9) ou servir só duas.

Há uma ressalva sobre esse resultado. As medidas do desafio.csv estão longe da faixa do wines.csv: o açúcar residual médio é 42,6 g/L contra 5,4, o dióxido de enxofre total é 256 contra 116 e a densidade média é 1,020 contra 0,995. O classificador de cor foi treinado em uma região dos dados diferente da que ele está prevendo, então as probabilidades devem ser lidas com cautela, em especial as duas garrafas com 0,78 e 0,63. Antes de servir, valeria confirmar a cor das duas por inspeção direta.

## 8. Conclusões

A planilha danificada foi recuperada com desvio de média menor que 0,03% em todas as colunas, e o texto adulterado foi tratado junto com os vazios pela coerção para numérico.

A regressão logística não serve para preencher uma variável contínua: transformar o teor alcoólico em faixas joga fora informação que não volta. A polinomial de grau 2 preencheu as 15 linhas com erro médio de 0,40 grau, contra 0,51 da logística e 1,09 da mediana. O KNNImputer empatou, o que confirma que o ganho vem de usar as outras variáveis, e o grau 3 mostrou o sobreajuste típico de polinômios grandes.

Na exploração, a densidade e o álcool têm a relação mais forte entre variáveis físico-químicas, e o álcool é a única variável com correlação acima de 0,4 com a nota. Os outliers de açúcar são quase todos brancos.

Na classificação, Random Forest e XGBoost chegaram a 91% de acurácia e AUC acima de 0,98, a otimização por Hyperopt não passou do patamar dos parâmetros padrão, e o Naive Bayes ficou em 55% por causa da dependência entre as variáveis. As duplicatas da base de classificação e a distância entre o desafio.csv e a base de treino são as duas limitações que mais pesam sobre esses números.

No desafio, só duas garrafas atendem à regra do crítico segundo o modelo, e a decisão sobre a terceira é da direção.

## 9. Reprodução

O notebook roda em Python 3.14 com pandas, numpy, scikit-learn, seaborn, matplotlib, plotly, statsmodels, xgboost, catboost e hyperopt. Os caminhos dos dados são relativos à pasta `notebooks/`. A busca de hiperparâmetros da seção 6.3 leva alguns minutos; o notebook guarda os parâmetros encontrados na primeira busca em uma célula própria para permitir pular essa etapa. Os outputs das células são limpos antes de cada commit, conforme o README.
