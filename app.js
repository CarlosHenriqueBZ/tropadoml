/* Apresentação interativa do trabalho. Os dados vêm de data/data.js (window.DATA),
   gerado por build_data.py a partir do notebook. Nada aqui recalcula modelo.
   Cada gráfico lê o tom do slide ou do card onde está (data-tone) para escolher as cores. */
(function () {
  const D = window.DATA;
  if (!D) {
    document.body.insertAdjacentHTML('afterbegin',
      '<p style="color:#fff;padding:2em;font-family:sans-serif">data/data.js não encontrado. Rode <code>python site/build_data.py</code>.</p>');
    return;
  }

  // paleta neutra, uma variante por tom de fundo
  const PAL = {
    dark: {
      text: '#f1efe8', muted: '#b8b6ad', grid: 'rgba(241,239,232,.10)', hoverBg: '#1e1f1c',
      a: '#b7cbb0', b: '#d9d2c3', c: '#7fa387', d: '#8f918a', e: '#8e9aa5', f: '#c2a58f',
      tinto: '#c9a0a0', branco: '#e9dfc4', destaque: '#f1efe8',
      seq: ['rgba(0,0,0,0)', '#3d5647', '#7fa387', '#b7cbb0', '#e6ede1'],
      caixa: ['#e6ede1', '#c9d6c2', '#a9c0a4', '#7fa387', '#5d7f6a', '#3d5647', '#2b3d33'],
    },
    light: {
      text: '#1e1f1c', muted: '#5f615b', grid: 'rgba(30,31,28,.09)', hoverBg: '#f4f3ee',
      a: '#3d5647', b: '#b3a78a', c: '#7fa387', d: '#8f918a', e: '#4f5e69', f: '#a67c60',
      tinto: '#7a5c5c', branco: '#c0b088', destaque: '#1e1f1c',
      seq: ['rgba(0,0,0,0)', '#e6ede1', '#b7cbb0', '#7fa387', '#3d5647'],
      caixa: ['#c9d6c2', '#a9c0a4', '#7fa387', '#5d7f6a', '#3d5647', '#2b3d33', '#1e2b24'],
    },
  };
  const DIVERGENTE = [[0, '#3d5647'], [0.5, '#e6e3d8'], [1, '#a67c60']];
  const CFG = { displayModeBar: false, responsive: true };
  const NOME_COR = { red: 'tinto', white: 'branco' };

  const fmt = (x, d = 2) => (x === null || x === undefined || isNaN(x)) ? ''
    : Number(x).toLocaleString('pt-BR', { minimumFractionDigits: d, maximumFractionDigits: d });

  function tone(el) {
    const t = el.closest('[data-tone]');
    return (t && t.dataset.tone) || 'dark';
  }
  function pal(el) { return PAL[tone(el)]; }

  function layout(el, extra = {}) {
    const P = pal(el);
    const ax = {
      gridcolor: P.grid, zerolinecolor: P.grid, linecolor: P.grid,
      tickfont: { size: 12, color: P.muted }, title: { font: { size: 13, color: P.muted } },
    };
    const base = {
      paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
      font: { family: 'Manrope, Inter, system-ui, sans-serif', color: P.text, size: 13 },
      margin: { l: 60, r: 20, t: 20, b: 50 },
      legend: { orientation: 'h', y: -0.22, x: 0, font: { size: 12, color: P.text } },
      hoverlabel: { bgcolor: P.hoverBg, bordercolor: P.a, font: { color: P.text, family: 'Manrope' } },
    };
    const out = Object.assign({}, base, extra);
    for (const k of Object.keys(out)) {
      if (/^[xy]axis\d*$/.test(k)) out[k] = Object.assign({}, ax, out[k]);
    }
    if (!out.xaxis) out.xaxis = Object.assign({}, ax);
    if (!out.yaxis) out.yaxis = Object.assign({}, ax);
    return out;
  }

  function tabela(el, cols, rows) {
    const th = cols.map(c => `<th class="${c.num ? 'num' : ''}">${c.label}</th>`).join('');
    const tr = rows.map(r => `<tr class="${r._cls || ''}">` +
      cols.map(c => `<td class="${c.num ? 'num' : ''}">${c.fmt ? c.fmt(r[c.k], r) : r[c.k]}</td>`).join('') +
      '</tr>').join('');
    el.innerHTML = `<table><thead><tr>${th}</tr></thead><tbody>${tr}</tbody></table>`;
  }

  function byId(id) { return document.getElementById(id); }

  // ------------------------------------------------------------------ gráficos
  const charts = {};

  charts['ch-mini-notas'] = el => {
    const P = pal(el), q = D.eda.quality_color, notas = D.eda.notas;
    Plotly.newPlot(el, [{
      type: 'bar', x: notas, y: notas.map(n => (q.red[n] || 0) + (q.white[n] || 0)), marker: { color: P.a },
      hovertemplate: 'nota %{x}: %{y} vinhos<extra></extra>',
    }], layout(el, { margin: { l: 4, r: 4, t: 4, b: 22 }, xaxis: { dtick: 1, showgrid: false, tickfont: { size: 10 } }, yaxis: { visible: false }, bargap: 0.25 }), CFG);
  };

  charts['ch-mini-cor'] = el => {
    const P = pal(el);
    Plotly.newPlot(el, [
      { type: 'bar', orientation: 'h', x: [D.meta.n_white], y: ['cor'], name: 'brancos', marker: { color: P.branco }, hovertemplate: '%{x} brancos<extra></extra>' },
      { type: 'bar', orientation: 'h', x: [D.meta.n_red], y: ['cor'], name: 'tintos', marker: { color: P.tinto }, hovertemplate: '%{x} tintos<extra></extra>' },
    ], layout(el, { barmode: 'stack', showlegend: true, legend: { orientation: 'h', y: -0.6, font: { size: 11 } }, margin: { l: 4, r: 4, t: 4, b: 4 }, xaxis: { visible: false }, yaxis: { visible: false }, bargap: 0.4 }), CFG);
  };

  charts['ch-problemas'] = el => {
    const P = pal(el), t = D.tratamento, cols = t.colunas;
    Plotly.newPlot(el, [
      { type: 'bar', name: 'células vazias', x: cols, y: cols.map(c => t.faltantes[c]), marker: { color: P.d } },
      { type: 'bar', name: 'texto no lugar do valor', x: cols, y: cols.map(c => t.incongruentes[c]), marker: { color: P.b } },
    ], layout(el, { barmode: 'stack', xaxis: { tickangle: -35 }, yaxis: { title: { text: 'células' } }, margin: { b: 110 } }), CFG);
  };

  charts['ch-validacao'] = el => {
    const P = pal(el), v = D.tratamento.validacao;
    Plotly.newPlot(el, [{
      type: 'bar', orientation: 'h', x: v.map(r => Math.abs(r.dif_pct)), y: v.map(r => r.col),
      text: v.map(r => fmt(r.dif_pct, 4) + '%'), textposition: 'outside', textfont: { size: 11, color: P.muted },
      marker: { color: P.a },
      hovertemplate: '%{y}<br>média no gabarito: %{customdata[0]}<br>média tratada: %{customdata[1]}<extra></extra>',
      customdata: v.map(r => [fmt(r.media_ref, 4), fmt(r.media_trat, 4)]),
    }], layout(el, { xaxis: { title: { text: 'diferença absoluta da média (%)' }, range: [0, 0.045] }, margin: { l: 150, r: 60 }, yaxis: { autorange: 'reversed' } }), CFG);
  };

  charts['ch-faixas'] = el => {
    const P = pal(el);
    Plotly.newPlot(el, [{
      type: 'histogram', x: D.wines.alcohol, nbinsx: 60, marker: { color: P.a },
      hovertemplate: 'álcool %{x}<br>%{y} vinhos<extra></extra>',
    }], layout(el, { xaxis: { title: { text: 'teor alcoólico (%)' } }, yaxis: { title: { text: 'vinhos' } }, shapes: [], annotations: [] }), CFG);
  };

  charts['ch-cm-log'] = el => {
    const P = pal(el), L = D.regressao.logistica, n = L.cm_teste, labels = D.regressao.faixas.nomes;
    Plotly.newPlot(el, [{
      type: 'heatmap', z: n, x: labels, y: labels, text: n, texttemplate: '%{text}', textfont: { size: 20, color: P.destaque },
      colorscale: [[0, 'rgba(127,127,127,0.06)'], [1, tone(el) === 'dark' ? '#5d7f6a' : '#b7cbb0']], showscale: false,
      hovertemplate: 'real: %{y}<br>prevista: %{x}<br>%{z} vinhos<extra></extra>',
    }], layout(el, { xaxis: { title: { text: 'faixa prevista' } }, yaxis: { title: { text: 'faixa real' }, autorange: 'reversed' }, margin: { l: 90, b: 60, t: 10 } }), CFG);
  };

  charts['ch-grau'] = el => {
    const P = pal(el), cv = D.regressao.poli.cv;
    Plotly.newPlot(el, [{
      type: 'bar', x: cv.map(c => `grau ${c.grau}`), y: cv.map(c => c.r2),
      error_y: { type: 'data', array: cv.map(c => c.std), color: P.muted, thickness: 1.2 },
      text: cv.map(c => `${c.termos} termos`), textposition: 'outside', textfont: { color: P.muted, size: 12 },
      marker: { color: [P.d, P.a, P.f] },
      hovertemplate: '%{x}<br>R² médio: %{y:.3f}<br>%{text}<extra></extra>',
    }], layout(el, { yaxis: { title: { text: 'R² médio na validação cruzada' }, range: [-0.3, 1.1], zeroline: true, zerolinecolor: P.muted }, margin: { t: 20 } }), CFG);
  };

  charts['ch-scatter-poli'] = el => {
    const P = pal(el), s = D.regressao.poli.scatter, p = D.regressao.poli.pior;
    Plotly.newPlot(el, [
      { type: 'scattergl', mode: 'markers', x: s.real, y: s.previsto, name: 'vinhos de teste',
        marker: { color: P.a, size: 5, opacity: 0.55 }, hovertemplate: 'real %{x}<br>previsto %{y:.2f}<extra></extra>' },
      { type: 'scatter', mode: 'lines', x: [7.5, 15], y: [7.5, 15], name: 'previsto = real', line: { color: P.muted, width: 1.5, dash: 'dot' }, hoverinfo: 'skip' },
      { type: 'scatter', mode: 'markers+text', x: [p.real], y: [p.previsto], name: 'pior erro', text: ['pior erro'], textposition: 'right', textfont: { color: P.text, size: 12 },
        marker: { color: P.f, size: 12, line: { color: P.text, width: 1.5 } },
        hovertemplate: `real ${fmt(p.real, 1)}, previsto ${fmt(p.previsto, 2)}<br>chlorides ${fmt(p.chlorides, 2)}, sulphates ${fmt(p.sulphates, 1)}<extra></extra>` },
    ], layout(el, { xaxis: { title: { text: 'álcool real' } }, yaxis: { title: { text: 'álcool previsto (grau 2)' } }, legend: { y: -0.18 } }), CFG);
  };

  charts['ch-15'] = el => {
    const P = pal(el), L = D.regressao.linhas15, x = L.map(r => String(r.idx));
    const serie = (k, nome, cor, vis) => ({
      type: 'bar', name: nome, x, y: L.map(r => r[k]), marker: { color: cor }, visible: vis,
      hovertemplate: `linha %{x}<br>${nome}: %{y:.2f}<extra></extra>`,
    });
    Plotly.newPlot(el, [
      serie('real', 'real (wines.csv)', P.destaque, true),
      serie('mediana', 'mediana da coluna', P.d, 'legendonly'),
      serie('logistica', 'logística', P.f, true),
      serie('polinomial', 'polinomial grau 2', P.a, true),
      serie('knn', 'KNNImputer', P.e, 'legendonly'),
    ], layout(el, { barmode: 'group', bargap: 0.25, yaxis: { title: { text: 'teor alcoólico' }, range: [7, 15] }, xaxis: { title: { text: 'linha do dataset' }, type: 'category' }, legend: { y: -0.25 } }), CFG);
    el.parentElement.querySelectorAll('.btns button').forEach(b => b.addEventListener('click', () => {
      const i = Number(b.dataset.trace);
      const vis = el.data[i].visible === true ? 'legendonly' : true;
      Plotly.restyle(el, { visible: vis }, [i]);
      b.classList.toggle('on', vis === true);
    }));
  };

  charts['ch-count-quality'] = el => {
    const P = pal(el), q = D.eda.quality_color, notas = D.eda.notas;
    const tot = { red: D.meta.n_red, white: D.meta.n_white };
    const trace = (c, modo) => ({
      type: 'bar', name: NOME_COR[c], x: notas, y: notas.map(n => modo === 'pct' ? 100 * (q[c][n] || 0) / tot[c] : (q[c][n] || 0)),
      marker: { color: P[NOME_COR[c]] },
      hovertemplate: modo === 'pct' ? `nota %{x}<br>${NOME_COR[c]}: %{y:.1f}%<extra></extra>` : `nota %{x}<br>${NOME_COR[c]}: %{y} vinhos<extra></extra>`,
    });
    const lay = modo => layout(el, { barmode: 'group', xaxis: { title: { text: 'nota de qualidade' }, dtick: 1 }, yaxis: { title: { text: modo === 'pct' ? '% dos vinhos da cor' : 'vinhos' } } });
    Plotly.newPlot(el, [trace('red', 'abs'), trace('white', 'abs')], lay('abs'), CFG);
    el.parentElement.querySelectorAll('.btns button').forEach(b => b.addEventListener('click', () => {
      const modo = b.dataset.modo;
      Plotly.animate(el, { data: [trace('red', modo), trace('white', modo)], layout: lay(modo) }, { transition: { duration: 600, easing: 'cubic-in-out' }, frame: { duration: 600 } });
      el.parentElement.querySelectorAll('.btns button').forEach(x => x.classList.toggle('on', x === b));
    }));
  };

  charts['ch-kde'] = el => {
    const P = pal(el), w = D.wines;
    const escala = P.seq.map((c, i) => [i / (P.seq.length - 1), c]);
    Plotly.newPlot(el, [
      { type: 'histogram2dcontour', x: w.alcohol, y: w.residual_sugar, colorscale: escala, showscale: false,
        contours: { coloring: 'heatmap' }, ncontours: 24, line: { width: 0 }, hovertemplate: 'álcool %{x}<br>açúcar %{y}<br>%{z} vinhos<extra></extra>' },
      { type: 'histogram', x: w.alcohol, yaxis: 'y2', marker: { color: P.c }, nbinsx: 40, hoverinfo: 'skip' },
      { type: 'histogram', y: w.residual_sugar, xaxis: 'x2', marker: { color: P.b }, nbinsy: 60, hoverinfo: 'skip' },
    ], layout(el, {
      showlegend: false, bargap: 0.05,
      xaxis: { domain: [0, 0.78], title: { text: 'álcool (%)' } },
      yaxis: { domain: [0, 0.78], title: { text: 'açúcar residual (g/L)' }, range: [0, 32] },
      xaxis2: { domain: [0.8, 1], showticklabels: false, showgrid: false },
      yaxis2: { domain: [0.8, 1], showticklabels: false, showgrid: false },
      margin: { t: 10 },
    }), CFG);
  };

  charts['ch-box'] = el => {
    const P = pal(el), w = D.wines, notas = D.eda.notas;
    const traces = notas.map((n, i) => ({
      type: 'box', name: String(n), y: w.residual_sugar.filter((_, k) => w.quality[k] === n),
      boxpoints: 'outliers', marker: { color: P.caixa[i % P.caixa.length], size: 3, opacity: 0.6 }, line: { color: P.caixa[i % P.caixa.length] },
      hovertemplate: 'nota %{x}<br>açúcar %{y}<extra></extra>',
    }));
    Plotly.newPlot(el, traces, layout(el, { showlegend: false, xaxis: { title: { text: 'nota de qualidade' } }, yaxis: { title: { text: 'açúcar residual (g/L)' } }, margin: { t: 10 } }), CFG);
  };

  charts['ch-outliers'] = el => {
    const P = pal(el), o = D.eda.outliers.por_quality, notas = D.eda.notas;
    Plotly.newPlot(el, ['red', 'white'].map(c => ({
      type: 'bar', name: NOME_COR[c], x: notas, y: notas.map(n => o[c][n] || 0), marker: { color: P[NOME_COR[c]] }, opacity: 0.85,
      hovertemplate: `nota %{x}<br>${NOME_COR[c]}: %{y} outliers<extra></extra>`,
    })), layout(el, { barmode: 'overlay', xaxis: { title: { text: 'nota de qualidade' }, dtick: 1 }, yaxis: { title: { text: 'outliers de açúcar residual' } }, margin: { t: 10 } }), CFG);
  };

  charts['ch-corr'] = el => {
    const P = pal(el), c = D.eda.corr;
    Plotly.newPlot(el, [{
      type: 'heatmap', z: c.matrix, x: c.cols, y: c.cols, zmin: -1, zmax: 1, colorscale: DIVERGENTE,
      text: c.matrix, texttemplate: '%{text:.2f}', textfont: { size: 10, color: '#1e1f1c' },
      colorbar: { thickness: 10, len: 0.8, tickfont: { color: P.muted } },
      hovertemplate: '%{y} × %{x}<br>r = %{z:.2f}<extra></extra>',
    }], layout(el, { xaxis: { tickangle: -40, tickfont: { size: 10 } }, yaxis: { tickfont: { size: 10 }, autorange: 'reversed' }, margin: { l: 130, b: 120, t: 10, r: 10 } }), CFG);
  };

  const PARES = {
    so2: { x: 'free_so2', y: 'total_so2', xl: 'SO2 livre (mg/L)', yl: 'SO2 total (mg/L)', ols: 'so2', xr: [0, 300] },
    dens: { x: 'density', y: 'alcohol', xl: 'densidade', yl: 'álcool (%)', ols: 'dens', xr: [0.985, 1.012] },
  };
  function scatterPar(el, key) {
    const P = pal(el), p = PARES[key], w = D.wines, ols = D.eda.ols[p.ols];
    const traces = [];
    for (const c of ['red', 'white']) {
      const idx = w.color.map((v, i) => v === c ? i : -1).filter(i => i >= 0);
      traces.push({ type: 'scattergl', mode: 'markers', name: NOME_COR[c], x: idx.map(i => w[p.x][i]), y: idx.map(i => w[p.y][i]),
        marker: { color: P[NOME_COR[c]], size: 5, opacity: 0.5 }, hovertemplate: `${p.xl}: %{x}<br>${p.yl}: %{y}<extra>${NOME_COR[c]}</extra>` });
    }
    for (const c of ['red', 'white']) {
      const xs = p.xr, o = ols[c];
      traces.push({ type: 'scatter', mode: 'lines', name: `tendência ${NOME_COR[c]}`, x: xs, y: xs.map(x => o.a * x + o.b), line: { color: P.text, width: c === 'red' ? 2 : 1.2, dash: c === 'red' ? 'solid' : 'dash' }, hoverinfo: 'skip' });
    }
    Plotly.react(el, traces, layout(el, { xaxis: { title: { text: p.xl }, range: p.xr }, yaxis: { title: { text: p.yl } }, legend: { y: -0.2 } }), CFG);
  }
  charts['ch-scatter'] = el => {
    scatterPar(el, 'so2');
    el.parentElement.querySelectorAll('.btns button').forEach(b => b.addEventListener('click', () => {
      scatterPar(el, b.dataset.par);
      el.parentElement.querySelectorAll('.btns button').forEach(x => x.classList.toggle('on', x === b));
      document.querySelectorAll('[data-par-texto]').forEach(t => { t.hidden = t.dataset.parTexto !== b.dataset.par; });
    }));
  };

  charts['ch-modelos'] = el => {
    const P = pal(el), r = D.classificacao.resultados;
    const met = [['acc', 'acurácia', P.destaque], ['rec', 'recall macro', P.f], ['f1', 'F1 macro', P.a], ['auc', 'ROC AUC macro', P.e]];
    const dados = zero => met.map(([k, nome, cor]) => ({
      type: 'bar', name: nome, x: r.map(m => m.modelo), y: r.map(m => zero ? 0 : m[k]), marker: { color: cor },
      hovertemplate: `%{x}<br>${nome}: %{y:.3f}<extra></extra>`,
    }));
    const lay = layout(el, { barmode: 'group', yaxis: { range: [0.4, 1.02], title: { text: `no teste (${D.classificacao.n_teste.toLocaleString('pt-BR')} vinhos)` } }, legend: { y: -0.2 } });
    Plotly.newPlot(el, dados(true), lay, CFG).then(() =>
      Plotly.animate(el, { data: dados(false) }, { transition: { duration: 900, easing: 'cubic-in-out' }, frame: { duration: 900 } }));
  };

  charts['ch-roc'] = el => {
    const P = pal(el), rocs = [...D.classificacao.roc].sort((a, b) => a.auc - b.auc);
    const cores = { KNN: P.e, 'Random Forest': P.a, CatBoost: P.c, AdaBoost: P.d, XGBoost: P.f, SVM: P.b };
    Plotly.newPlot(el, [{ type: 'scatter', mode: 'lines', x: [0, 1], y: [0, 1], name: 'acaso (AUC 0,50)', line: { color: P.muted, dash: 'dash', width: 1 }, hoverinfo: 'skip' }],
      layout(el, { xaxis: { title: { text: 'taxa de falsos positivos' }, range: [0, 1] }, yaxis: { title: { text: 'taxa de verdadeiros positivos' }, range: [0, 1.02] }, legend: { orientation: 'v', x: 0.55, y: 0.05, font: { size: 11 } } }), CFG);
    rocs.forEach((r, i) => setTimeout(() => { if (el.data) Plotly.addTraces(el, {
      type: 'scatter', mode: 'lines', x: r.fpr, y: r.tpr, name: `${r.modelo} (AUC ${fmt(r.auc, 3)})`,
      line: { color: cores[r.modelo] || P.text, width: r.auc > 0.98 ? 3 : 2 }, hovertemplate: `${r.modelo}<br>FPR %{x:.2f}, TPR %{y:.2f}<extra></extra>`,
    }); }, 350 * (i + 1)));
  };

  charts['ch-imp'] = el => {
    const P = pal(el), imp = D.classificacao.importancias;
    Plotly.newPlot(el, [
      { type: 'bar', orientation: 'h', name: 'Random Forest', y: imp.map(i => i.feature), x: imp.map(i => i.rf), marker: { color: P.a } },
      { type: 'bar', orientation: 'h', name: 'XGBoost', y: imp.map(i => i.feature), x: imp.map(i => i.xgb), marker: { color: P.f } },
    ], layout(el, { barmode: 'group', xaxis: { title: { text: 'importância' }, tickformat: '.0%' }, margin: { l: 160, t: 10 }, legend: { y: -0.15 } }), CFG);
  };

  charts['ch-corr-classe'] = el => {
    const P = pal(el), cc = D.classificacao.corr_por_classe;
    Plotly.newPlot(el, [
      { type: 'bar', name: 'r > 0,4', x: cc.map(c => `nota ${c.nota}`), y: cc.map(c => c.positivas), marker: { color: P.a } },
      { type: 'bar', name: 'r < -0,4', x: cc.map(c => `nota ${c.nota}`), y: cc.map(c => c.negativas), marker: { color: P.f } },
    ], layout(el, { barmode: 'stack', yaxis: { title: { text: `pares fortes (de ${cc[0].pares})` } }, margin: { t: 10 }, legend: { y: -0.25 } }), CFG);
  };

  charts['ch-medias'] = el => {
    const P = pal(el), m = D.desafio.medias;
    Plotly.newPlot(el, [
      { type: 'bar', name: 'wines.csv (treino)', x: m.map(r => r.col), y: m.map(r => r.wines), marker: { color: P.a } },
      { type: 'bar', name: 'desafio.csv', x: m.map(r => r.col), y: m.map(r => r.desafio), marker: { color: P.f } },
    ], layout(el, { barmode: 'group', yaxis: { type: 'log', title: { text: 'média (escala log)' } }, xaxis: { tickangle: -35 }, margin: { b: 110, t: 10 }, legend: { y: -0.45 } }), CFG);
  };

  // passos acionados por fragments (data-step="nome:n"); idempotentes, para poder ressincronizar
  const steps = {
    'faixas:1': on => {
      const el = byId('ch-faixas'); if (!el || !el.data) return;
      const P = pal(el), f = D.regressao.faixas;
      const ann = (el.layout.annotations || []).filter(a => a.mediana);
      const rot = [
        { x: (f.limites[0] + f.limites[1]) / 2, y: 1.02, yref: 'paper', text: 'baixo', showarrow: false, font: { color: P.text } },
        { x: (f.limites[1] + f.limites[2]) / 2, y: 1.02, yref: 'paper', text: 'médio', showarrow: false, font: { color: P.text } },
        { x: (f.limites[2] + f.limites[3]) / 2, y: 1.02, yref: 'paper', text: 'alto', showarrow: false, font: { color: P.text } },
      ];
      Plotly.relayout(el, {
        shapes: on ? [f.limites[1], f.limites[2]].map(x => ({ type: 'line', x0: x, x1: x, y0: 0, y1: 1, yref: 'paper', line: { color: P.f, width: 2, dash: 'dash' } })) : [],
        annotations: on ? rot.concat(ann) : ann,
      });
    },
    'faixas:2': on => {
      const el = byId('ch-faixas'); if (!el || !el.data) return;
      const P = pal(el), f = D.regressao.faixas;
      const ann = (el.layout.annotations || []).filter(a => !a.mediana);
      if (on) f.nomes.forEach(n => ann.push({ mediana: true, x: f.medianas[n], y: 0.5, yref: 'paper', text: `mediana ${fmt(f.medianas[n], 1)}`, showarrow: true, arrowhead: 2, arrowcolor: P.text, ax: 0, ay: -40, font: { color: P.text, size: 12 } }));
      Plotly.relayout(el, { annotations: ann });
    },
  };

  // tabelas e textos montados a partir dos dados
  const inits = {};

  inits['log-metricas'] = () => {
    const L = D.regressao.logistica.teste;
    tabela(byId('tb-log'), [{ k: 'm', label: 'métrica' }, { k: 'v', label: 'teste', num: true }], [
      { m: 'acurácia', v: fmt(L.acc, 3) }, { m: 'precisão (macro)', v: fmt(L.prec, 3) }, { m: 'recall (macro)', v: fmt(L.rec, 3) },
      { m: 'F1 (macro)', v: fmt(L.f1, 3) }, { m: 'ROC AUC (um contra o resto)', v: fmt(L.auc, 3) },
    ]);
  };

  inits['poli-metricas'] = () => {
    const t = D.regressao.poli.teste;
    byId('kp-mae').firstChild.textContent = fmt(t.MAE, 2);
    byId('kp-rmse').firstChild.textContent = fmt(t.RMSE, 2);
    byId('kp-r2').firstChild.textContent = fmt(t.R2, 3);
    byId('kp-mape').firstChild.textContent = fmt(t.MAPE, 1) + '%';
  };

  inits['resumo15'] = () => {
    const nomes = { mediana: 'mediana da coluna', logistica: 'logística em 3 faixas', polinomial: 'polinomial grau 2', knn: 'KNNImputer (k = 5)' };
    tabela(byId('tb-resumo'), [
      { k: 'tecnica', label: 'técnica', fmt: v => nomes[v] }, { k: 'MAE', label: 'MAE', num: true, fmt: v => fmt(v, 3) },
      { k: 'RMSE', label: 'RMSE', num: true, fmt: v => fmt(v, 3) }, { k: 'R2', label: 'R²', num: true, fmt: v => fmt(v, 3) },
    ], D.regressao.resumo.map(r => Object.assign({ _cls: r.tecnica === 'polinomial' ? 'destaque' : '' }, r)));
  };

  inits['pares'] = () => {
    tabela(byId('tb-pares'), [
      { k: 'v1', label: 'variável' }, { k: 'v2', label: 'variável' }, { k: 'r', label: 'r', num: true, fmt: v => fmt(v, 2) },
    ], D.eda.pares_fortes);
  };

  inits['modelos'] = () => {
    tabela(byId('tb-modelos'), [
      { k: 'modelo', label: 'modelo' }, { k: 'acc', label: 'acurácia', num: true, fmt: v => fmt(v, 3) },
      { k: 'prec', label: 'precisão', num: true, fmt: v => fmt(v, 3) }, { k: 'rec', label: 'recall', num: true, fmt: v => fmt(v, 3) },
      { k: 'f1', label: 'F1', num: true, fmt: v => fmt(v, 3) }, { k: 'auc', label: 'ROC AUC', num: true, fmt: v => fmt(v, 3) },
    ], D.classificacao.resultados.map(r => Object.assign({ _cls: r.f1 > 0.9 ? 'destaque' : '' }, r)));
  };

  inits['otimizacao'] = () => {
    const b = D.classificacao.rf_otimizado;
    const nomeP = { criterion: 'critério', max_depth: 'profundidade', max_features: 'variáveis por divisão', min_samples_leaf: 'mín. por folha', min_samples_split: 'mín. por divisão', n_estimators: 'árvores', C: 'C', gamma: 'gamma' };
    const val = v => v === null ? 'sem limite' : (typeof v === 'number' && !Number.isInteger(v) ? fmt(v, 3) : v);
    b.forEach((s, i) => {
      const el = byId(`busca-${i + 1}`);
      if (!el) return;
      el.innerHTML = `<h3>${s.modelo}</h3><p class="tiny" style="color:inherit;opacity:.75">${Object.entries(s.params).map(([k, v]) => `${nomeP[k] || k}: <b>${val(v)}</b>`).join(' · ')}</p>` +
        `<p class="small">recall macro na validação cruzada: <span class="hl">${fmt(s.recall_cv, 3)}</span><br>` +
        `teste: acurácia ${fmt(s.teste.acc, 3)}, F1 macro ${fmt(s.teste.f1, 3)}, AUC ${fmt(s.teste.auc, 3)}</p>`;
    });
    const nb = D.classificacao.nb;
    byId('nb-acc').firstChild.textContent = fmt(nb.acc, 2);
    byId('nb-f1').firstChild.textContent = fmt(nb.f1, 2);
  };

  inits['onze'] = () => {
    tabela(byId('tb-onze'), [
      { k: 'quality', label: 'nota', num: true }, { k: 'cor_prevista', label: 'cor prevista' },
      { k: 'prob_tinto', label: 'prob. tinto', num: true, fmt: v => fmt(v, 2) }, { k: 'alcohol', label: 'álcool', num: true, fmt: v => fmt(v, 1) },
      { k: 'residual_sugar', label: 'açúcar', num: true, fmt: v => fmt(v, 1) }, { k: 'sulphates', label: 'sulfatos', num: true, fmt: v => fmt(v, 2) },
    ], D.desafio.onze.map(r => Object.assign({ _cls: r.cor_prevista === 'tinto' ? 'tinto' : 'apagado' }, r)));
  };

  inits['cv-cor'] = () => {
    tabela(byId('tb-cv'), [
      { k: 'modelo', label: 'modelo' }, { k: 'f1_treino', label: 'F1 treino', num: true, fmt: v => fmt(v, 4) },
      { k: 'f1_val', label: 'F1 validação', num: true, fmt: v => fmt(v, 4) }, { k: 'dif_f1', label: 'diferença', num: true, fmt: v => fmt(v, 4) },
      { k: 'prec_tinto_val', label: 'precisão tinto', num: true, fmt: v => fmt(v, 4) },
    ], D.desafio.cv_cor.map(r => Object.assign({ _cls: r.modelo === D.desafio.escolhido ? 'destaque' : '' }, r)));
  };

  // contadores animados: <span class="count" data-to="6497" data-dec="0">
  function countUp(el) {
    if (el.dataset.done) return;
    el.dataset.done = '1';
    const to = parseFloat(el.dataset.to), dec = parseInt(el.dataset.dec || '0', 10), suf = el.dataset.suf || '';
    const dur = 1100, t0 = performance.now();
    const step = t => {
      const p = Math.min(1, (t - t0) / dur), e = 1 - Math.pow(1 - p, 3);
      el.textContent = fmt(to * e, dec) + suf;
      if (p < 1) requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
  }

  function render(el) {
    const fn = charts[el.id];
    if (!fn) return;
    if (el.dataset.rendered) { try { Plotly.Plots.resize(el); } catch (e) { /* ainda sem plot */ } return; }
    el.dataset.rendered = '1';
    try { fn(el); } catch (e) { console.error('gráfico', el.id, e); }
  }

  function sincronizaPassos(slide) {
    slide.querySelectorAll('[data-step]').forEach(f => {
      const s = f.dataset.step;
      if (steps[s]) steps[s](f.classList.contains('visible'));
    });
  }

  function onSlide(slide) {
    if (!slide) return;
    slide.querySelectorAll('[data-chart]').forEach(render);
    slide.querySelectorAll('.count').forEach(countUp);
    const init = slide.dataset.init;
    if (init && inits[init] && !slide.dataset.initDone) {
      slide.dataset.initDone = '1';
      try { inits[init](slide); } catch (e) { console.error('init', init, e); }
    }
    setTimeout(() => sincronizaPassos(slide), 80);
  }

  // rodapé de cada slide: marca à esquerda, número à direita
  function rodape() {
    const secoes = document.querySelectorAll('.reveal .slides > section');
    secoes.forEach((s, i) => {
      s.insertAdjacentHTML('beforeend',
        `<div class="foot"><svg class="mark" viewBox="0 0 24 24" aria-hidden="true"><rect x="2" y="12" width="10" height="10" rx="3"/><rect x="12" y="2" width="10" height="10" rx="3"/></svg><span>${i + 1} / ${secoes.length}</span></div>`);
    });
  }

  // ------------------------------------------------------------------ reveal
  rodape();
  Reveal.initialize({
    hash: true, width: 1280, height: 720, margin: 0.04,
    transition: 'slide', backgroundTransition: 'fade', transitionSpeed: 'default',
    controls: true, progress: true, slideNumber: false, center: false,
    plugins: [RevealNotes],
  });
  Reveal.on('ready', e => onSlide(e.currentSlide));
  Reveal.on('slidechanged', e => onSlide(e.currentSlide));
  Reveal.on('fragmentshown', e => { const s = e.fragment.dataset.step; if (s && steps[s]) steps[s](true); });
  Reveal.on('fragmenthidden', e => { const s = e.fragment.dataset.step; if (s && steps[s]) steps[s](false); });
  window.addEventListener('resize', () => document.querySelectorAll('.present [data-chart][data-rendered]').forEach(el => { try { Plotly.Plots.resize(el); } catch (e) { /* ignora */ } }));
})();
