"""Gera site/data/data.js com os numeros e os pontos usados na apresentacao.

Repete os passos do notebook (tratamento, regressao, EDA, classificacao e desafio)
com as mesmas sementes, para que o site mostre os mesmos numeros do relatorio.
A busca de hiperparametros nao e refeita: os parametros encontrados nas duas buscas
ficam fixos em BUSCAS_HYPEROPT e so o ajuste final de cada Random Forest e reexecutado.

Uso: python site/build_data.py   (leva cerca de um minuto)
"""

import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.base import clone
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.impute import KNNImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (accuracy_score, auc, confusion_matrix, f1_score, make_scorer,
                             mean_absolute_error, mean_absolute_percentage_error,
                             mean_squared_error, precision_score, r2_score, recall_score,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import (StratifiedKFold, cross_val_score, cross_validate,
                                     train_test_split)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (LabelEncoder, PolynomialFeatures, StandardScaler,
                                   label_binarize)
from sklearn.svm import SVC
from xgboost import XGBClassifier

RAIZ = Path(__file__).resolve().parents[1]
DADOS = RAIZ / 'data'
SAIDA = Path(__file__).resolve().parent / 'data' / 'data.js'

# parametros encontrados pelo Hyperopt no notebook (50 avaliacoes, recall macro em CV de 5 partes)
BUSCAS_HYPEROPT = [
    {'busca': 1, 'recall_cv': 0.832, 'scaler_no_pipeline': False,
     'params': {'criterion': 'entropy', 'max_depth': 24, 'max_features': 'log2',
                'min_samples_leaf': 1, 'min_samples_split': 6, 'n_estimators': 200}},
    {'busca': 2, 'recall_cv': 0.834, 'scaler_no_pipeline': True,
     'params': {'criterion': 'gini', 'max_depth': 30, 'max_features': 'log2',
                'min_samples_leaf': 1, 'min_samples_split': 4, 'n_estimators': 450}},
]


def limpa(o):
    """Converte tipos numpy/pandas para JSON e arredonda floats."""
    if isinstance(o, dict):
        return {str(k): limpa(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [limpa(v) for v in o]
    if isinstance(o, (pd.Series, pd.Index, np.ndarray)):
        return limpa(o.tolist())
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        return None if np.isnan(o) else round(float(o), 5)
    return o


def metricas_cls(y, p):
    return {'acc': accuracy_score(y, p),
            'prec': precision_score(y, p, average='macro', zero_division=0),
            'rec': recall_score(y, p, average='macro', zero_division=0),
            'f1': f1_score(y, p, average='macro', zero_division=0)}


def metricas_reg(y, p):
    return {'MAE': mean_absolute_error(y, p), 'MSE': mean_squared_error(y, p),
            'RMSE': float(np.sqrt(mean_squared_error(y, p))),
            'MAPE': mean_absolute_percentage_error(y, p) * 100, 'R2': r2_score(y, p)}


D = {}

# ---------------------------------------------------------------- tratamento
pre = pd.read_csv(DADOS / 'wines_pre_processing.csv')
wines = pd.read_csv(DADOS / 'wines.csv')

faltantes = pre.isnull().sum()
colunas_num = pre.columns.drop('color')
incong = {}
for col in colunas_num:
    conv = pd.to_numeric(pre[col], errors='coerce')
    mask = conv.isna() & pre[col].notna()
    incong[col] = pre.loc[mask, col].astype(str).tolist()
    pre[col] = conv
cores_inval = ~pre['color'].isin(['white', 'red']) & pre['color'].notna()
incong['color'] = pre.loc[cores_inval, 'color'].astype(str).tolist()

continuas = colunas_num.drop(['quality', 'alcohol'])
for col in continuas:
    pre[col] = pre[col].fillna(pre[col].median())
pre.loc[cores_inval, 'color'] = np.nan
pre['quality'] = pre['quality'].fillna(pre['quality'].mode()[0]).astype(int)
pre['color'] = pre['color'].fillna(pre['color'].mode()[0])

colunas = [c for c in wines.columns]
cols_val = [c for c in colunas if c != 'color']
D['tratamento'] = {
    'colunas': colunas,
    'faltantes': {c: int(faltantes[c]) for c in colunas},
    'incongruentes': {c: len(incong.get(c, [])) for c in colunas},
    'textos': sorted({t for c, v in incong.items() if c != 'quality' for t in v}),
    'validacao': [{'col': c, 'media_ref': wines[c].mean(), 'media_trat': pre[c].mean(),
                   'dif_pct': 100 * (pre[c].mean() - wines[c].mean()) / wines[c].mean(),
                   'std_ref': wines[c].std(), 'std_trat': pre[c].std()} for c in cols_val],
    'color_ref': wines['color'].value_counts().to_dict(),
    'color_trat': pre['color'].value_counts().to_dict(),
    'quality_ref': wines['quality'].value_counts().sort_index().to_dict(),
    'quality_trat': pre['quality'].value_counts().sort_index().to_dict(),
    'total_faltantes': int(faltantes.drop('Unnamed: 0').sum()),
    'total_incongruentes': sum(len(v) for v in incong.values()),
    'alcohol_faltantes': int(pre['alcohol'].isna().sum()),
    'sem_cor_valida': int(cores_inval.sum() + faltantes['color']),
}

# ---------------------------------------------------------------- regressao (alcohol)
df_reg = pre.drop(columns=['Unnamed: 0']).copy()
df_reg['color'] = (df_reg['color'] == 'red').astype(int)
falta = df_reg['alcohol'].isna()
X = df_reg.drop(columns=['alcohol'])
Xk, Xf = X[~falta], X[falta]
yk = df_reg.loc[~falta, 'alcohol']
real = wines.loc[falta, 'alcohol']

nomes = ['baixo', 'medio', 'alto']
yfx, limites = pd.qcut(yk, q=3, labels=nomes, retbins=True)
med_faixa = yk.groupby(yfx, observed=True).median()

Xt, Xte, yt, yte = train_test_split(Xk, yfx, test_size=0.2, random_state=42, stratify=yfx)
log = Pipeline([('escala', StandardScaler()), ('logistica', LogisticRegression(max_iter=2000))]).fit(Xt, yt)
fp, pp = log.predict(Xte), log.predict_proba(Xte)
por_faixa = {}
for f in nomes:
    m = (yte == f)
    por_faixa[f] = {'precision': precision_score(yte == f, fp == f, zero_division=0),
                    'recall': recall_score(yte == f, fp == f, zero_division=0),
                    'f1': f1_score(yte == f, fp == f, zero_division=0), 'support': int(m.sum())}

ff, pf = log.predict(Xf), log.predict_proba(Xf)
alc_log = pd.Series(med_faixa.loc[ff].values, index=Xf.index)
faixa_real = pd.cut(real, bins=limites, labels=nomes, include_lowest=True)

Xt_r, Xte_r, yt_r, yte_r = train_test_split(Xk, yk, test_size=0.2, random_state=42)


def poli(grau):
    return Pipeline([('escala', StandardScaler()),
                     ('poli', PolynomialFeatures(degree=grau, include_bias=False)),
                     ('linear', LinearRegression())])


cv_graus = []
for g in (1, 2, 3):
    sc = cross_val_score(poli(g), Xt_r, yt_r, cv=5, scoring='r2')
    termos = poli(g).fit(Xt_r, yt_r)['poli'].n_output_features_
    cv_graus.append({'grau': g, 'termos': int(termos), 'r2': sc.mean(), 'std': sc.std()})
melhor = max(cv_graus, key=lambda d: d['r2'])['grau']
mp = poli(melhor).fit(Xt_r, yt_r)
prev_teste = mp.predict(Xte_r)
prev_s = pd.Series(prev_teste, index=yte_r.index)
pior = (prev_s - yte_r).abs().idxmax()
alc_poli = pd.Series(mp.predict(Xf), index=Xf.index)

esc = StandardScaler().fit(Xk)
Xkn = pd.DataFrame(esc.transform(X), columns=X.columns, index=X.index)
Xkn['alcohol'] = df_reg['alcohol']
alc_knn = pd.Series(KNNImputer(n_neighbors=5, weights='distance').fit_transform(Xkn)[:, -1],
                    index=X.index).loc[Xf.index]

comp = pd.DataFrame({'real': real, 'mediana': yk.median(), 'logistica': alc_log,
                     'polinomial': alc_poli.round(2), 'knn': alc_knn.round(2)})
comp['faixa_real'] = faixa_real.astype(str)
comp['faixa_prevista'] = ff

D['regressao'] = {
    'faixas': {'nomes': nomes, 'limites': limites, 'medianas': med_faixa.to_dict(),
               'contagem': yfx.value_counts().to_dict()},
    'corr_alcohol': df_reg[~falta].corr()['alcohol'].drop('alcohol').sort_values().to_dict(),
    'logistica': {
        'teste': {**metricas_cls(yte, fp), 'auc': roc_auc_score(yte, pp, multi_class='ovr'), 'n': int(len(yte))},
        'por_faixa': por_faixa,
        'cm_teste': confusion_matrix(yte, fp, labels=nomes),
        'val15': {**metricas_cls(faixa_real, ff), 'auc': roc_auc_score(faixa_real, pf, multi_class='ovr'),
                  'mae': mean_absolute_error(real, alc_log),
                  'rmse': float(np.sqrt(mean_squared_error(real, alc_log))),
                  'cm': confusion_matrix(faixa_real, ff, labels=nomes)},
    },
    'poli': {
        'cv': cv_graus, 'melhor_grau': melhor,
        'teste': {**metricas_reg(yte_r, prev_teste), 'n': int(len(yte_r))},
        'scatter': {'real': yte_r.values, 'previsto': prev_teste},
        'pior': {'idx': int(pior), 'real': yte_r[pior], 'previsto': prev_s[pior],
                 'chlorides': Xte_r.loc[pior, 'chlorides'], 'sulphates': Xte_r.loc[pior, 'sulphates']},
        'val15': metricas_reg(real, alc_poli),
    },
    'linhas15': [{'idx': int(i), **{k: r[k] for k in comp.columns}} for i, r in comp.iterrows()],
    'resumo': [{'tecnica': t, **metricas_reg(real, comp[t])} for t in ('mediana', 'logistica', 'polinomial', 'knn')],
}

# ---------------------------------------------------------------- EDA (wines.csv)
w = wines
D['wines'] = {'color': w['color'], 'quality': w['quality'], 'alcohol': w['alcohol'],
              'residual_sugar': w['residual sugar'], 'density': w['density'],
              'free_so2': w['free sulfur dioxide'], 'total_so2': w['total sulfur dioxide']}

q1 = w.groupby('quality')['residual sugar'].transform(lambda s: s.quantile(0.25))
q3 = w.groupby('quality')['residual sugar'].transform(lambda s: s.quantile(0.75))
iqr = q3 - q1
eh_out = (w['residual sugar'] < q1 - 1.5 * iqr) | (w['residual sugar'] > q3 + 1.5 * iqr)
outl = w[eh_out]
ct_out = pd.crosstab(outl['quality'], outl['color'])
corr = w.select_dtypes('number').corr()
mask = np.triu(np.ones(corr.shape), k=1).astype(bool)
pares = corr.where(mask).stack().dropna().reset_index()
pares.columns = ['v1', 'v2', 'r']
fortes = pares[(pares['r'] > 0.4) | (pares['r'] < -0.4)].sort_values('r', ascending=False)


def ols(x, y):
    a, b = np.polyfit(x, y, 1)
    return {'a': a, 'b': b}


ct_q = pd.crosstab(w['quality'], w['color'])
D['eda'] = {
    'notas': sorted(w['quality'].unique()),
    'quality_color': {c: ct_q[c].to_dict() for c in ('red', 'white')},
    'outliers': {'total': int(eh_out.sum()), 'por_cor': outl['color'].value_counts().to_dict(),
                 'por_quality': {c: (ct_out[c].to_dict() if c in ct_out else {}) for c in ('red', 'white')},
                 'maior': {'valor': w['residual sugar'].max(),
                           **w.loc[w['residual sugar'].idxmax(), ['color', 'quality']].to_dict()}},
    'corr': {'cols': corr.columns.tolist(), 'matrix': corr.values},
    'pares_fortes': fortes.to_dict(orient='records'),
    'ols': {'so2': {c: ols(w.loc[w.color == c, 'free sulfur dioxide'], w.loc[w.color == c, 'total sulfur dioxide']) for c in ('red', 'white')},
            'dens': {c: ols(w.loc[w.color == c, 'density'], w.loc[w.color == c, 'alcohol']) for c in ('red', 'white')}},
    'sugar_lt5_pct': 100 * (w['residual sugar'] < 5).mean(),
    'sugar_gt20_n': int((w['residual sugar'] > 20).sum()),
    'alcohol_9_125_pct': 100 * w['alcohol'].between(9, 12.5).mean(),
    'sugar_mediana_cor': w.groupby('color')['residual sugar'].median().to_dict(),
}

# ---------------------------------------------------------------- classificacao da qualidade
dc = pd.read_csv(DADOS / 'wine_classification.csv')
enc = LabelEncoder()
yq = enc.fit_transform(dc['quality'])
Xq = dc.drop(columns=['quality', 'level_0', 'index'])
Xq_tr, Xq_te, yq_tr, yq_te = train_test_split(Xq, yq, test_size=0.2, random_state=12, stratify=yq)
esc_q = StandardScaler().set_output(transform='pandas')
Xq_tr_e, Xq_te_e = esc_q.fit_transform(Xq_tr), esc_q.transform(Xq_te)

modelos_q = [
    ('KNN', KNeighborsClassifier()),
    ('Random Forest', RandomForestClassifier(n_estimators=100, random_state=12, n_jobs=-1)),
    ('CatBoost', CatBoostClassifier(n_estimators=100, loss_function='MultiClass', verbose=0, random_state=12)),
    ('AdaBoost', AdaBoostClassifier(n_estimators=100, random_state=12)),
    ('XGBoost', XGBClassifier(n_estimators=100, eval_metric='mlogloss', random_state=12)),
    ('SVM', SVC()),
]
grid = np.linspace(0, 1, 101)
resultados, roc_out, importancias = [], [], {}
for nome, m in modelos_q:
    m.fit(Xq_tr_e, yq_tr)
    p = np.ravel(m.predict(Xq_te_e)).astype(int)
    proba = m.predict_proba(Xq_te_e) if hasattr(m, 'predict_proba') else m.decision_function(Xq_te_e)
    ybin = label_binarize(yq_te, classes=m.classes_)
    aucs, tprs = [], []
    for i in range(len(m.classes_)):
        fpr, tpr, _ = roc_curve(ybin[:, i], proba[:, i])
        aucs.append(auc(fpr, tpr))
        tprs.append(np.interp(grid, fpr, tpr))
    auc_macro = float(np.mean(aucs))
    resultados.append({'modelo': nome, **metricas_cls(yq_te, p), 'auc': auc_macro})
    roc_out.append({'modelo': nome, 'auc': auc_macro, 'fpr': grid, 'tpr': np.mean(tprs, axis=0)})
    if isinstance(m, (RandomForestClassifier, XGBClassifier)):
        importancias[nome] = m.feature_importances_

imp = pd.DataFrame(importancias, index=Xq_tr.columns)
imp = imp.loc[imp.mean(axis=1).sort_values().index]

rf_otim = []
for b in BUSCAS_HYPEROPT:
    if b['scaler_no_pipeline']:
        rf = Pipeline([('scaler', StandardScaler()),
                       ('modelo', RandomForestClassifier(**b['params'], random_state=12, n_jobs=-1))]).fit(Xq_tr, yq_tr)
        p, pr = rf.predict(Xq_te), rf.predict_proba(Xq_te)
    else:
        rf = RandomForestClassifier(**b['params'], random_state=12, n_jobs=-1).fit(Xq_tr_e, yq_tr)
        p, pr = rf.predict(Xq_te_e), rf.predict_proba(Xq_te_e)
    rf_otim.append({'busca': b['busca'], 'params': b['params'], 'recall_cv': b['recall_cv'],
                    'teste': {**metricas_cls(yq_te, p), 'auc': roc_auc_score(yq_te, pr, multi_class='ovr')}})

nb = GaussianNB().fit(Xq_tr_e, yq_tr)
p_nb = nb.predict(Xq_te_e)

df_an = Xq_tr.copy()
df_an['classe'] = yq_tr
corr_classe = []
for cl in sorted(np.unique(yq_tr)):
    c = df_an[df_an['classe'] == cl].drop(columns='classe').corr().values
    pares_cl = c[np.triu_indices(c.shape[0], 1)]
    pos, neg = int((pares_cl > 0.4).sum()), int((pares_cl < -0.4).sum())
    corr_classe.append({'classe': int(cl), 'nota': int(enc.classes_[cl]), 'positivas': pos,
                        'negativas': neg, 'total': pos + neg, 'pares': int(len(pares_cl))})

D['classificacao'] = {
    'n': int(len(dc)), 'n_treino': int(len(Xq_tr)), 'n_teste': int(len(Xq_te)),
    'dist_quality': dc['quality'].astype(int).value_counts().sort_index().to_dict(),
    'duplicadas': int(Xq.assign(q=dc['quality']).duplicated().sum()),
    'classes': [int(c) for c in enc.classes_],
    'resultados': resultados, 'roc': roc_out,
    'importancias': [{'feature': f, 'rf': imp.loc[f, 'Random Forest'], 'xgb': imp.loc[f, 'XGBoost']} for f in imp.index],
    'rf_otimizado': rf_otim,
    'nb': {**metricas_cls(yq_te, p_nb), 'cm': confusion_matrix(yq_te, p_nb)},
    'corr_por_classe': corr_classe,
}

# ---------------------------------------------------------------- desafio (cor + nota 8 ou 9)
dd = pd.read_csv(DADOS / 'desafio.csv')
w2 = wines.copy()
w2['quality'] = w2['quality'].astype(float)
w2 = pd.get_dummies(w2, columns=['color'], drop_first=True, dtype=float)
y2 = w2['color_white']
X2 = w2.drop(columns=['color_white', 'quality'])
X2_tr, X2_te, y2_tr, y2_te = train_test_split(X2, y2, test_size=0.2, random_state=12, stratify=y2)
esc2 = StandardScaler().set_output(transform='pandas')
X2_tr_e, X2_te_e = esc2.fit_transform(X2_tr), esc2.transform(X2_te)

modelos_c = [
    ('KNN', KNeighborsClassifier()),
    ('Random Forest', RandomForestClassifier(n_estimators=100, random_state=12, n_jobs=-1)),
    ('CatBoost', CatBoostClassifier(n_estimators=100, loss_function='Logloss', verbose=0, random_state=12)),
    ('AdaBoost', AdaBoostClassifier(n_estimators=100, random_state=12)),
    ('XGBoost', XGBClassifier(n_estimators=100, eval_metric='logloss', random_state=12)),
    ('SVM', SVC()),
]
modelos_cor = []
for nome, m in modelos_c:
    m.fit(X2_tr_e, y2_tr)
    p = np.ravel(m.predict(X2_te_e)).astype(int)
    modelos_cor.append({'modelo': nome, 'acc': accuracy_score(y2_te, p),
                        'f1': f1_score(y2_te, p, average='macro'),
                        'prec_tinto': precision_score(y2_te, p, pos_label=0, zero_division=0),
                        'cm': confusion_matrix(y2_te, p)})

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=12)
scoring = {'acuracia': 'accuracy', 'f1_macro': 'f1_macro',
           'precisao_tinto': make_scorer(precision_score, pos_label=0, zero_division=0)}
cv_cor = []
for nome, m in modelos_c:
    pipe = Pipeline([('scaler', StandardScaler()), ('modelo', clone(m))])
    sc = cross_validate(pipe, X2_tr, y2_tr, cv=cv, scoring=scoring, return_train_score=True)
    linha = {'modelo': nome, 'acc_treino': sc['train_acuracia'].mean(), 'acc_val': sc['test_acuracia'].mean(),
             'f1_treino': sc['train_f1_macro'].mean(), 'f1_val': sc['test_f1_macro'].mean(),
             'desvio_f1': sc['test_f1_macro'].std(), 'prec_tinto_val': sc['test_precisao_tinto'].mean()}
    linha['dif_f1'] = linha['f1_treino'] - linha['f1_val']
    cv_cor.append(linha)

xgb = next(m for n, m in modelos_c if n == 'XGBoost')
pipe_cor = Pipeline([('scaler', StandardScaler()), ('modelo', clone(xgb))]).fit(X2_tr, y2_tr)
feats = X2_tr.columns.tolist()
Xd = dd.loc[:, feats]
dd['cor_prevista'] = pd.Series(pipe_cor.predict(Xd), index=dd.index).map({0: 'tinto', 1: 'branco'})
dd['prob_tinto'] = pipe_cor.predict_proba(Xd)[:, list(pipe_cor.classes_).index(0)]
onze = dd[dd['quality'] >= 8].sort_values(['quality', 'prob_tinto'], ascending=[False, False])

D['desafio'] = {
    'n': int(len(dd)),
    'dist_quality': dd['quality'].astype(int).value_counts().sort_index().to_dict(),
    'modelos_cor': modelos_cor, 'cv_cor': cv_cor, 'escolhido': 'XGBoost',
    'cor_prevista': dd['cor_prevista'].value_counts().to_dict(),
    'onze': [{'quality': int(r['quality']), 'cor_prevista': r['cor_prevista'], 'prob_tinto': r['prob_tinto'],
              'alcohol': r['alcohol'], 'residual_sugar': r['residual sugar'], 'sulphates': r['sulphates'],
              'volatile_acidity': r['volatile acidity'], 'density': r['density']} for _, r in onze.iterrows()],
    'nota7': {'tintos': int(((dd['quality'] == 7) & (dd['cor_prevista'] == 'tinto')).sum()),
              'prob09': int(((dd['quality'] == 7) & (dd['prob_tinto'] > 0.9)).sum())},
    'medias': [{'col': c, 'desafio': dd[c].mean(), 'wines': X2[c].mean()} for c in feats],
}

D['meta'] = {'gerado_em': date.today().isoformat(), 'n_wines': int(len(wines)),
             'n_white': int((wines['color'] == 'white').sum()), 'n_red': int((wines['color'] == 'red').sum()),
             'n_class': int(len(dc)), 'n_desafio': int(len(dd))}

SAIDA.parent.mkdir(parents=True, exist_ok=True)
conteudo = json.dumps(limpa(D), ensure_ascii=False, separators=(',', ':'))
SAIDA.write_text('// gerado por site/build_data.py, nao editar a mao\nwindow.DATA = ' + conteudo + ';\n', encoding='utf-8')
print(f'ok: {SAIDA} ({SAIDA.stat().st_size / 1024:.0f} KB)')
