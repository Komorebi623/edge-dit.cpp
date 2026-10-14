#!/usr/bin/env node
'use strict';
/**
 * Editable H3 review, schema_version=1. No measurements are inferred from reports.
 * Usage: node scripts/build_h3_review_deck.js --data DATA.json --output OUT.pptx
 * Assets resolve relative to DATA.json; missing measurements/assets stay pending.
 * Additional AdaLN fields: samples_s, dit_samples_s, profile_allocation_s,
 * profile_labels, profile_segments. Media may be a path or {path,name,poster}.
 * Only OUT.pptx is written. Every drawable is bounds-checked and named H3:*.
 */
const fs = require('fs');
const path = require('path');
const W = 13.333333, H = 7.5, M = 0.5;
const FONT = 'Noto Sans CJK SC';
const C = { ink: '18232B', teal: '157A7E', orange: 'C17C28', white: 'FFFFFF',
  gray: '56636A', line: 'D8E1E2', light: 'EDF5F5', warm: 'FAF3E9' };
const ROOT = path.resolve(__dirname, '..');
const REPORTS = {
  cache: 'h3-four-method-results.md', subblock: 'h3-subblock-keep25.md',
  adaln: 'h3-adaln-fix.md', lora: 'h3-lora-results.md', overnight: 'h3-vsa-native-status.md',
};
const SOURCE_NAMES = { cache: '四方法对照记录', subblock: 'SubBlock 修复与保留率对照',
  adaln: 'AdaLN 修复与回归记录', lora: 'LoRA 实测与部署记录', overnight: '夜间任务状态 / VSA 实验记录' };
const num = x => typeof x === 'number' && Number.isFinite(x);
const fmt = (x, d = 2) => num(x) ? x.toFixed(d) : '待测';
const short = (x, n = 64) => { const s = String(x == null ? '未提供' : x); return [...s].length > n ? [...s].slice(0, n - 1).join('') + '…' : s; };
function validateData(d) {
  if (!d || d.schema_version !== 1) throw new Error('DATA.schema_version must be 1');
  for (const key of ['setup', 'results', 'assets', 'media', 'overnight', 'sources']) {
    if (d[key] != null && (typeof d[key] !== 'object' || Array.isArray(d[key]))) throw new Error(`${key} must be an object`);
  }
  for (const [key, r] of Object.entries(d.results || {})) {
    if (!r || typeof r !== 'object' || Array.isArray(r)) throw new Error(`results.${key} must be an object`);
    for (const k of ['cli_s', 'dit_s', 'base_cli_s', 'ssim', 'audio_rmse', 'n']) {
      if (r[k] != null && (!num(r[k]) || (k === 'ssim' ? r[k] < -1 || r[k] > 1 : r[k] < 0))) throw new Error(`Invalid results.${key}.${k}`);
    }
    if (r.n != null && (!Number.isInteger(r.n) || r.n < 1)) throw new Error(`Invalid results.${key}.n`);
    for (const k of ['samples_s', 'dit_samples_s', 'profile_allocation_s']) {
      if (r[k] != null && (!Array.isArray(r[k]) || r[k].some(x => x != null && (!num(x) || x < 0)))) throw new Error(`Invalid results.${key}.${k}`);
    }
  }
  if (d.overnight?.tasks != null && !Array.isArray(d.overnight.tasks)) throw new Error('overnight.tasks must be an array');
  return d;
}
function loadPptx() {
  try { return require('pptxgenjs'); } catch (e) {
    if (e.code !== 'MODULE_NOT_FOUND') throw e;
    throw new Error('Missing pptxgenjs. Source _scratch/h3-report-runtime/env.sh (renderer only); do not change the model runtime.');
  }
}
class Drawing {
  constructor(pptx, slide, index) { this.pptx = pptx; this.slide = slide; this.index = index; this.bounds = []; }
  track(kind, x, y, w, h, name = '') {
    if (![x, y, w, h].every(num) || w < 0 || h < 0 || x < M - 0.005 || y < M - 0.005 || x + w > W - M + 0.005 || y + h > H - M + 0.005) {
      throw new Error(`Slide ${this.index} ${kind}/${name}: out of safe bounds (${x},${y},${w},${h})`);
    }
    const objectName = `H3:${kind}:${this.bounds.length + 1}:${short(name, 42)}`;
    this.bounds.push({ kind, x, y, w, h, name: objectName }); return objectName;
  }
  text(text, x, y, w, h, opt = {}) {
    const fontSize = opt.fontSize || 18;
    if (fontSize < 16 && !opt.footnote) throw new Error('Body text must be at least 16pt');
    const objectName = this.track(opt.footnote ? 'footer' : 'text', x, y, w, h, String(text));
    const clean = { ...opt }; delete clean.footnote;
    this.slide.addText(String(text), { x, y, w, h, fontFace: FONT, fontSize,
      color: C.ink, margin: 0, breakLine: false, valign: 'mid', paraSpaceAfterPt: 0,
      lang: 'zh-CN', objectName, ...clean });
  }
  shape(type, x, y, w, h, opt = {}) {
    const objectName = this.track('shape', x, y, w, h, type);
    this.slide.addShape(type, { x, y, w, h, objectName, line: { color: C.line, width: 1 }, fill: { color: C.white }, ...opt });
  }
  box(text, x, y, w, h, opt = {}) {
    const { fill = C.light, color = C.ink, fontSize = 18, ...rest } = opt;
    this.shape('rect', x, y, w, h, { fill: { color: fill }, line: { color: C.line, width: 1 } });
    this.text(text, x + 0.12, y + 0.08, w - 0.24, h - 0.16, { align: 'center', color, fontSize, ...rest });
  }
  line(x1, y1, x2, y2, arrow = true, color = C.teal, dash = false) {
    const x = Math.min(x1, x2), y = Math.min(y1, y2), w = Math.abs(x2 - x1), h = Math.abs(y2 - y1);
    const objectName = this.track('line', x, y, w, h, 'connector');
    this.slide.addShape('line', { x, y, w, h, flipH: x2 < x1, flipV: y2 < y1, objectName,
      line: { color, width: 1.6, ...(arrow ? { beginArrowType: 'none', endArrowType: 'triangle' } : {}), ...(dash ? { dashType: 'dash' } : {}) } });
  }
  chart(title, labels, values, x, y, w, h) {
    const pairs = labels.map((label, i) => [label, values[i]]).filter(p => num(p[1]));
    if (!pairs.length) { this.box(`${title}\n待测 / 未提供数据`, x, y, w, h, { fill: C.white, color: C.gray }); return; }
    const objectName = this.track('chart', x, y, w, h, title);
    this.slide.addChart(this.pptx.ChartType?.bar || 'bar', [{ name: '秒', labels: pairs.map(p => p[0]), values: pairs.map(p => p[1]) }], {
      x, y, w, h, objectName, barDir: 'bar', barGrouping: 'clustered',
      showTitle: true, title, titleFontFace: FONT, titleFontSize: 18, titleColor: C.ink,
      showLegend: false, showValue: true, showCatName: false, showBorder: false,
      chartColors: [C.gray, C.orange, C.teal, C.teal],
      catAxisLabelFontFace: FONT, catAxisLabelFontSize: 16, catAxisLabelColor: C.ink,
      valAxisLabelFontFace: FONT, valAxisLabelFontSize: 12, valAxisLabelColor: C.gray,
      dataLabelFormatCode: '0.00', dataLabelPosition: 'outEnd', dataLabelBkgrdColor: C.white,
      dataLabelColor: C.ink, dataLabelFontFace: FONT, dataLabelFontSize: 16,
      valAxisMinVal: 0, valAxisMaxVal: Math.max(...pairs.map(p => p[1]), 1) * 1.24,
      catAxisLineShow: false, valAxisLineShow: false,
      catGridLine: { style: 'none' }, valGridLine: { color: C.line, width: 0.5 },
      showShadow: false, showMarker: false, gapSize: 70,
      showPlotArea: true, showChartArea: true, chartArea: { fill: { color: C.white }, border: { color: C.white, pt: 0 } },
      plotArea: { fill: { color: C.white }, border: { color: C.white, pt: 0 } },
    });
    if (pairs.length !== labels.length) this.text('缺失项未绘制', x + w - 1.7, y + h - 0.2, 1.65, 0.2, { footnote: true, fontSize: 11, color: C.gray, align: 'right' });
  }
  image(file, x, y, w, h, caption = '对照抽帧待提供') {
    if (!file || !fs.existsSync(file)) { this.box(caption, x, y, w, h, { fill: C.white, color: C.gray }); return false; }
    const b = fs.readFileSync(file);
    if (b.length < 24 || b.subarray(0, 8).toString('hex') !== '89504e470d0a1a0a') throw new Error(`Expected PNG asset: ${file}`);
    const iw = b.readUInt32BE(16), ih = b.readUInt32BE(20);
    if (!iw || !ih) throw new Error(`Invalid PNG dimensions: ${file}`);
    const scale = Math.min(w / iw, h / ih), dw = iw * scale, dh = ih * scale;
    const dx = x + (w - dw) / 2, dy = y + (h - dh) / 2;
    const objectName = this.track('image', dx, dy, dw, dh, path.basename(file));
    this.slide.addImage({ path: file, x: dx, y: dy, w: dw, h: dh, objectName }); return true;
  }
  video(media, x, y, w, h, name, resolve) {
    const file = resolve(typeof media === 'string' ? media : media?.path);
    const label = short(typeof media === 'object' && media?.name ? media.name : `${name} · ${file ? path.basename(file) : '待提供'}`, 38);
    this.text(label, x, y, w, 0.4, { fontSize: 16, bold: true });
    if (!file || !fs.existsSync(file)) { this.box('视频待提供', x, y + 0.48, w, h - 0.48, { fill: C.white, color: C.gray }); return; }
    const objectName = this.track('media', x, y + 0.48, w, h - 0.48, label);
    const poster = resolve(typeof media === 'object' ? media.poster : null);
    const cover = poster && fs.existsSync(poster) ? `image/png;base64,${fs.readFileSync(poster).toString('base64')}` : undefined;
    this.slide.addMedia({ type: 'video', path: file, extn: path.extname(file).slice(1), x, y: y + 0.48, w, h: h - 0.48, objectName, ...(cover ? { cover } : {}) });
  }
}
function buildDeck(data, dataFile, PptxGenJS = loadPptx()) {
  validateData(data);
  const pptx = new PptxGenJS();
  pptx.layout = 'LAYOUT_WIDE';
  pptx.author = 'H3 实验报告'; pptx.subject = '可编辑机制图、原生实测图表与产物对照';
  pptx.title = 'H3 单卡推理优化：机制、实测与产物'; pptx.company = 'edge-dit.cpp';
  pptx.lang = 'zh-CN'; pptx.theme = { headFontFace: FONT, bodyFontFace: FONT, lang: 'zh-CN' };
  const results = data.results || {}, r = key => results[key] || {};
  const resolve = file => {
    if (!file) return null;
    if (typeof file !== 'string' || /^(?:https?|data):/i.test(file)) throw new Error('Only local asset/media paths are allowed');
    return path.resolve(path.dirname(dataFile), file);
  };
  const asset = key => resolve(data.assets?.[key]);
  const drawings = [];
  function slide(title, subtitle, sources = [], extra = {}) {
    const s = pptx.addSlide(); s.background = { color: C.white };
    const d = new Drawing(pptx, s, drawings.length + 1); drawings.push(d);
    d.text(title, M, M, 12.2, 0.58, { fontSize: 32, bold: true });
    d.text(subtitle, M, 1.18, 12.25, 0.6, { fontSize: 18, color: C.gray });
    const footer = sources.map(k => SOURCE_NAMES[k] || k).join('；');
    d.text(`${short(footer || '主控数据与复现记录', 65)} · 口径/路径见备注`, M, 6.75, 10.6, 0.22, { fontSize: 10.5, footnote: true, color: C.gray });
    d.text(`${short(data.as_of || '时间未提供', 22)}  |  ${String(drawings.length).padStart(2, '0')}`, 10.7, 6.75, 2.1, 0.22, { fontSize: 10, footnote: true, align: 'right', color: C.gray });
    d.note = { title, sources, extra }; return d;
  }
  const speed = key => { const v = r(key); return num(v.base_cli_s) && num(v.cli_s) && v.cli_s > 0 ? `${(v.base_cli_s / v.cli_s).toFixed(2)}×` : '待测'; };
  const quality = key => `SSIM ${fmt(r(key).ssim, 4)}   音频 RMSE ${fmt(r(key).audio_rmse, 5)}`;
  // 01: concise findings, not a title-only filler page.
  {
    const d = slide('H3 单卡推理优化：机制、实测与产物', `${short(data.status_label || '阶段报告', 24)}  ·  四条独立路径，不把收益连乘`, ['cache', 'subblock', 'adaln', 'lora']);
    const cards = [
      ['CacheDiT', speed('cache'), '跳过中段；执行器收益另计'],
      ['SubBlock', speed('subblock25'), 'keep25；速度与保真取舍'],
      ['AdaLN', (r('adaln').samples_s || []).filter(num).length ? 'mixed 已测' : '待测', '移出投影；不跳 DiT 层'],
      ['LoRA', speed('larry4'), '4 步学生；改变生成轨迹'],
    ];
    cards.forEach((a, i) => { const x = 0.5 + i * 3.12; d.text(a[0], x, 2.12, 2.8, 0.45, { fontSize: 22, bold: true }); d.text(a[1], x, 2.82, 2.8, 0.7, { fontSize: 32, bold: true, color: C.teal }); d.text(a[2], x, 3.74, 2.78, 1.0, { fontSize: 18 }); });
    d.box('先拆分工程收益与算法收益，再比较时间和产物。', 0.5, 5.16, 12.3, 0.7, { fill: C.light, fontSize: 22 });
    d.text('加速比各自对应本组 CLI baseline；AdaLN 使用独立 mixed 输入。', 0.5, 6.02, 12.1, 0.42, { fontSize: 16, color: C.gray });
  }
  // 02: setup is entirely supplied by the controller.
  {
    const d = slide('受控设置：相同输入，分组比较', 'T2VA 与 mixed 分开；LoRA 的学生模型与采样变化显式标注', ['cache', 'subblock', 'lora'], { setup: data.setup });
    const setup = data.setup || {};
    [['硬件 / 输出', `${setup.gpu || '待提供'}\n${setup.geometry || '待提供'}`], ['精度 / 放置', `${setup.precision || '待提供'}\n${setup.placement || '待提供'}`], ['采样 / 原始基线', `${setup.sampling || '待提供'}\n${setup.baseline || '待提供'}`]].forEach((a, i) => {
      const y = 2.00 + i * 1.23; d.text(a[0], 0.5, y, 2.0, 0.6, { bold: true, fontSize: 18 }); d.text(short(a[1], 135), 2.55, y, 6.0, 1.08, { fontSize: 16, valign: 'top' });
    });
    d.box('T2VA\n文本 → 视频 + 音频', 9.0, 2.0, 3.8, 1.15);
    d.box('mixed\n图像 + 视频 + 额外音频', 9.0, 3.53, 3.8, 1.2, { fill: C.warm });
    d.text('独立 baseline\n不跨输入求加速比', 9.15, 5.1, 3.5, 0.85, { fontSize: 18, color: C.orange, align: 'center' });
    d.text(short(setup.controls || '输入、权重、采样、计时范围待主控补充', 105), 0.5, 6.12, 12.25, 0.45, { fontSize: 16, color: C.gray });
  }
  // 03: native residual-cache diagram.
  {
    const d = slide('CacheDiT：复用同锚中段残差，而非整份 latent', '锚步 k 走完整路径；步 m 先探测，命中后仍执行真实后段', ['cache']);
    d.text('锚步 k', 0.5, 2.22, 1.2, 0.5, { bold: true });
    d.box('前 2 层\nfront(k)', 2.0, 2.05, 2.25, 0.95); d.box('中间 46 层\nmiddle(k)', 5.05, 2.05, 3.25, 0.95); d.box('后 2 层\nback(k)', 9.4, 2.05, 2.7, 0.95);
    d.line(4.3, 2.52, 5.0, 2.52); d.line(8.35, 2.52, 9.35, 2.52);
    d.text('保存 Δk = middle(k) − front(k)', 4.65, 3.22, 4.2, 0.55, { color: C.teal, fontSize: 20, align: 'center' });
    d.line(6.65, 3.02, 6.65, 3.2, false);
    d.text('探测步 m', 0.5, 4.18, 1.4, 0.5, { bold: true });
    d.box('前 2 层\n残差探测 + 模态保护', 2.0, 3.98, 2.8, 1.13, { fontSize: 16 });
    d.box('命中：front(m) + Δk', 5.45, 3.98, 3.35, 1.13, { fontSize: 18 });
    d.box('真实后 2 层\n输出更新', 9.4, 3.98, 2.7, 1.13);
    d.line(4.85, 4.55, 5.4, 4.55); d.line(8.85, 4.55, 9.35, 4.55);
    d.line(6.65, 3.82, 6.65, 3.96, true, C.teal, true);
    d.text('未命中 → 完整中段计算，更新锚点', 3.2, 5.55, 8.3, 0.55, { color: C.orange, fontSize: 20, align: 'center' });
    d.text('跳过的是中段计算及相应参数传输；不是冻结所有模态输出。', 1.0, 6.16, 11.6, 0.4, { fontSize: 16, color: C.gray, align: 'center' });
  }
  // 04: cache attribution with its own experimental baseline.
  {
    const a = r('cache'), e = r('cache_executor');
    const d = slide('CacheDiT：总收益不能全部记为“缓存命中”', '同组 CLI 对照：原始 baseline → 执行器禁复用 → 启用残差复用', ['cache'], { cache: a, cache_executor: e });
    d.chart('T2VA CLI / 秒（越低越好）', ['原始 baseline', '执行器 / 禁复用', 'CacheDiT'], [a.base_cli_s, e.cli_s, a.cli_s], 0.5, 2.0, 7.4, 3.8);
    const delta1 = num(a.base_cli_s) && num(e.cli_s) ? a.base_cli_s - e.cli_s : null;
    const delta2 = num(e.cli_s) && num(a.cli_s) && e.base_cli_s === a.base_cli_s ? e.cli_s - a.cli_s : null;
    d.text('① 执行器部分', 8.35, 2.1, 4.4, 0.42, { bold: true, fontSize: 20 });
    d.text(`${fmt(delta1)} 秒`, 8.35, 2.72, 4.4, 0.65, { fontSize: 30, color: C.teal });
    d.text('② 额外残差复用部分', 8.35, 3.64, 4.4, 0.42, { bold: true, fontSize: 20 });
    d.text(`${fmt(delta2)} 秒`, 8.35, 4.26, 4.4, 0.65, { fontSize: 30, color: C.orange });
    d.text(`命中 ${num(a.hits) ? a.hits : '待提供'} / ${num(a.steps) ? a.steps : '待提供'} 步；n=${num(a.n) ? a.n : '待提供'}，不同于单次实验组`, 0.5, 6.0, 12.1, 0.5, { fontSize: 18 });
  }
  // 05: paired frames.
  {
    const d = slide('CacheDiT：同帧产物对照', quality('cache'), ['cache'], { asset: data.assets?.['cache_compare.png'], cache: r('cache') });
    d.image(asset('cache_compare.png'), 0.5, 1.98, 12.3, 3.88);
    d.text('残差近似可能改变纹理与运动；低绝对音频误差不等于听感无损。', 0.65, 6.0, 12.0, 0.54, { fontSize: 18 });
  }
  // 06: true per-head subcell routing; native vector grid, not bitmap.
  {
    const d = slide('SubBlock：用子块 LSE 路由 K，不删除 Q', 'Q64 / K64；每侧 4 个 16-token 子块 → 4×4 分数 → 每 head / 每 Q 块 TopK', ['subblock']);
    d.text('Q 子块 × K 子块', 0.55, 2.05, 3.2, 0.45, { bold: true, fontSize: 20 });
    for (let q = 0; q < 4; q++) for (let k = 0; k < 4; k++) d.box('q·k', 0.65 + k * 0.66, 2.72 + q * 0.66, 0.6, 0.6, { fontSize: 16, fill: q === k ? C.light : C.white });
    d.line(3.4, 4.0, 4.03, 4.0);
    d.box('LSE 汇聚\n每对 Q64 / K64', 4.1, 3.4, 2.35, 1.2, { fontSize: 18 }); d.line(6.5, 4.0, 7.13, 4.0);
    d.box('逐 head / Q 块\nTopK 选 K', 7.2, 3.4, 2.45, 1.2, { fontSize: 18 }); d.line(9.7, 4.0, 10.25, 4.0);
    d.box('索引注意力\n所有 Q 更新', 10.3, 3.4, 2.5, 1.2, { fontSize: 18 });
    d.text('保护非视频 Q / K / V；保留首末层、前 2 步和末步。', 0.6, 5.67, 12.1, 0.47, { fontSize: 20, color: C.teal });
    d.text('keep25 指目标视频 K 块名义保留率；不是整个注意力计算仅剩 25%。', 0.6, 6.18, 12.1, 0.4, { fontSize: 16, color: C.gray });
  }
  // 07: paired speed-quality points and frames.
  {
    const d = slide('SubBlock：keep25 的增益与保真代价', '相同修复版；只改变保留率。单次观察，不把 K 块减半理解为整模型快一倍', ['subblock'], { subblock50: r('subblock50'), subblock25: r('subblock25') });
    d.chart('T2VA CLI / 秒', ['本组 baseline', 'keep50', 'keep25'], [r('subblock25').base_cli_s, r('subblock50').cli_s, r('subblock25').cli_s], 0.5, 2.0, 5.2, 3.45);
    d.image(asset('subblock_compare.png'), 6.0, 2.0, 6.8, 3.48);
    d.text(`keep50  ${quality('subblock50')}`, 0.5, 5.66, 12.2, 0.4, { fontSize: 18 });
    d.text(`keep25  ${quality('subblock25')}`, 0.5, 6.18, 12.2, 0.4, { fontSize: 18, color: C.orange });
  }
  // 08: prefill table native diagram.
  {
    const d = slide('AdaLN：把时间投影移出重复热循环', '只缓存调制参数；注意力、MLP 与全部 DiT 层仍执行，不是 DiT 跳层', ['adaln']);
    d.text('请求预填充', 0.5, 2.13, 2.1, 0.45, { fontSize: 22, bold: true });
    d.box('time / σ', 0.6, 2.98, 1.7, 0.9); d.line(2.35, 3.43, 2.78, 3.43);
    d.box('共享 time embedding', 2.85, 2.85, 3.0, 1.15); d.line(5.9, 3.43, 6.38, 3.43);
    d.box('逐层投影', 6.45, 2.85, 2.3, 1.15); d.line(8.8, 3.43, 9.25, 3.43);
    d.box('表 [σ, layer]\nscale / shift / gate', 9.32, 2.7, 3.45, 1.5, { fontSize: 18 });
    d.line(11.02, 4.25, 11.02, 4.88);
    d.box('每步每层：查表 → 调制 → 原 attention / MLP → residual', 2.25, 4.95, 10.52, 0.85, { fill: C.white, fontSize: 20 });
    d.text('采样热循环', 0.5, 5.07, 1.6, 0.54, { fontSize: 18, bold: true });
    d.text('工程关键：共享 temb、缓存重入、边界感知图段合并；不能只数投影 FLOPs。', 0.5, 6.16, 12.25, 0.4, { fontSize: 16, color: C.gray });
  }
  // 09: strictly separate diagnostic profile and formal mixed samples.
  {
    const a = r('adaln'), samples = Array.isArray(a.samples_s) ? a.samples_s : [];
    const d = slide('AdaLN：修复分配退化，再看正式 mixed 计时', '左：同 instrumentation 的阶段诊断；右：正式 CLI。两种口径不能混算', ['adaln'], { adaln: a });
    d.chart('图 / 缓冲分配阶段 / 秒 · profile', a.profile_labels || ['baseline', '旧 AdaLN', '修复版'], a.profile_allocation_s || [], 0.5, 2.0, 6.0, 3.28);
    const labels = ['baseline 均值', ...samples.map((_, i) => `修复 / 第 ${i + 1} 次`)];
    const values = [a.base_cli_s, ...samples];
    if (!samples.length && num(a.cli_s)) { labels.push('修复版'); values.push(a.cli_s); }
    d.chart('mixed 正式 CLI / 秒', labels, values, 6.85, 2.0, 5.95, 3.28);
    d.image(asset('adaln_compare.png'), 0.5, 5.42, 5.75, 1.14, 'AdaLN 产物对照待提供');
    d.text(samples.some(num) ? '已完成样本逐个匹配某个 baseline 产物；\n不宣称按序三次验收完成。' : '正式样本待提供；\n不宣称重复验收完成。', 6.85, 5.46, 5.85, 1.05, { fontSize: 18, color: C.orange });
  }
  // 10: parallel low-rank branch, not destructive Q4 merge.
  {
    const d = slide('LoRA：保留 Q4 主干，旁路叠加 A / B', 'y = WQ4 x + α·B(Ax)；BF16 因子独立加载，不覆写或重新量化原模型', ['lora']);
    d.box('x', 0.6, 3.1, 1.0, 0.9, { fontSize: 24 });
    d.line(1.66, 3.55, 2.0, 3.55, false); d.line(2.0, 2.57, 2.0, 4.55, false);
    d.line(2.0, 2.57, 3.0, 2.57); d.box('原 Q4 线性层 W', 3.06, 2.07, 5.05, 1.0, { fontSize: 22 });
    d.line(2.0, 4.55, 3.0, 4.55); d.box('A · BF16', 3.06, 4.08, 2.0, 0.94); d.line(5.12, 4.55, 5.67, 4.55); d.box('B · BF16', 5.73, 4.08, 2.38, 0.94);
    d.line(8.17, 2.57, 9.4, 2.57, false); d.line(9.4, 2.57, 9.4, 3.34);
    d.line(8.17, 4.55, 9.4, 4.55, false); d.line(9.4, 4.55, 9.4, 3.93);
    d.box('+', 9.02, 3.36, 0.78, 0.56, { fontSize: 24, fill: C.warm }); d.line(9.87, 3.64, 10.5, 3.64); d.box('y', 10.57, 3.19, 1.8, 0.9, { fontSize: 24 });
    d.text('学生配方：20 → 4 次 DiT 调用', 0.7, 5.7, 11.9, 0.55, { fontSize: 26, color: C.teal, align: 'center' });
    d.text('速度变化同时包含学生模型与采样器变化，不是同权重的无损工程加速。', 0.5, 6.24, 12.25, 0.35, { fontSize: 16, color: C.gray });
  }
  // 11: native bar chart and editable table.
  {
    const d = slide('LoRA：少步学生明显缩短本次 CLI', '原 20 步 baseline 与两个 4 步候选；每个候选单次，不冒充稳定统计结论', ['lora'], { baseline: r('baseline_t2va'), larry4: r('larry4'), light4: r('light4') });
    d.chart('本组 T2VA CLI / 秒', ['原 20 步', 'Larry · 4 步', 'Light FL · 4 步'], [r('larry4').base_cli_s, r('larry4').cli_s, r('light4').cli_s], 0.5, 2.0, 6.05, 3.95);
    const rows = [['模型', 'CLI / s', 'DiT / s'], ['baseline', fmt(r('baseline_t2va').cli_s), fmt(r('baseline_t2va').dit_s)], ['Larry4', fmt(r('larry4').cli_s), fmt(r('larry4').dit_s)], ['Light4', fmt(r('light4').cli_s), fmt(r('light4').dit_s)]];
    rows.forEach((row, i) => row.forEach((v, j) => { const xs = [6.95, 9.05, 10.98], widths = [1.95, 1.8, 1.8]; d.text(v, xs[j], 2.32 + i * 0.7, widths[j], 0.48, { fontSize: 18, bold: i === 0, color: i === 0 ? C.teal : C.ink }); }));
    d.text('CLI 包含加载与封装；\n不是 load-once 服务稳态。', 6.95, 5.43, 5.8, 0.75, { fontSize: 18 });
    d.text('SSIM 是对基线的距离，不是画质保留率；需结合抽帧与完整音画判断。', 0.5, 6.24, 12.25, 0.35, { fontSize: 16, color: C.gray });
  }
  // 12: native playable media objects, one per candidate, no remote fetch.
  {
    const d = slide('LoRA 产物：同帧比较 + 可播放候选', '左侧抽帧仅展示局部；右侧视频嵌入文件，PowerPoint 放映时点击播放', ['lora'], { media: data.media, larry4: r('larry4'), light4: r('light4') });
    d.image(asset('lora_compare.png'), 0.5, 1.98, 8.12, 3.9);
    d.video(data.media?.larry4, 9.0, 1.98, 3.8, 2.12, 'Larry4', resolve);
    d.video(data.media?.light4, 9.0, 4.3, 3.8, 2.12, 'Light4', resolve);
    d.text(`Larry  SSIM ${fmt(r('larry4').ssim, 4)}  ·  Light  SSIM ${fmt(r('light4').ssim, 4)}`, 0.5, 6.02, 8.1, 0.5, { fontSize: 18 });
  }
  // 13: pending is never represented as a measured zero.
  {
    const overnight = data.overnight || {}, tasks = overnight.tasks || [];
    const d = slide('夜间接续：实验分支与验收状态', 'VSA 保持独立：fine + gate × compressed；不能借 LoRA 成绩宣称 VSA 已通过', ['overnight'], { overnight });
    d.box('fine attention', 0.5, 2.18, 3.05, 0.8); d.box('compressed attention', 0.5, 3.52, 3.05, 0.8, { fontSize: 16 });
    d.box('逐元素乘 gate(h)\ngate(h) = Wg · h', 4.14, 3.44, 2.65, 0.98, { fontSize: 16, fill: C.warm });
    d.line(3.61, 3.92, 4.08, 3.92); d.line(6.85, 3.92, 7.35, 3.92, false); d.line(7.35, 3.92, 7.35, 3.29);
    d.line(3.61, 2.58, 7.35, 2.58, false); d.line(7.35, 2.58, 7.35, 2.78); d.box('+', 7.0, 2.8, 0.7, 0.47, { fontSize: 20 });
    d.line(7.76, 3.04, 8.3, 3.04); d.box('输出', 8.36, 2.66, 1.35, 0.76);
    d.text('实验路径\nGPU 验收见状态表', 10.07, 2.48, 2.7, 1.15, { fontSize: 18, color: C.orange });
    const statusName = s => ({ pending: '待执行', not_started: '未启动', running: '执行中', blocked: '受阻 / 待 GPU', completed: '已完成', complete: '已完成', passed: '已通过', succeeded: '已通过', deadline: '截止未测', timeout: '超时停止', waiting_gpu: '待空闲 GPU', failed: '失败', done: '已完成', skipped: '未执行' }[s] || short(s || '待验收', 18));
    if (!tasks.length) d.box(`主控状态：${statusName(overnight.status)}\n暂无新增验收结果；GPU 时间 / 质量均待测。`, 0.5, 4.95, 12.3, 1.24, { fontSize: 20, fill: C.white });
    else {
      tasks.slice(0, 4).forEach((task, i) => {
        d.text(short(task.label || task.id || '任务', 30), 0.5, 4.65 + i * 0.48, 5.0, 0.46, { fontSize: 18 });
        d.text(statusName(task.status), 5.7, 4.65 + i * 0.48, 2.35, 0.46, { fontSize: 18, color: C.orange });
        const result = typeof task.result === 'string' ? task.result : num(task.result?.cli_s) ? `${fmt(task.result.cli_s)} 秒 / ${task.result.NFE || '?'} NFE` : task.kind === 'parity' && task.status === 'succeeded' ? 'CUDA 数值检查通过' : '无新增实测';
        d.text(short(result, 19), 8.3, 4.65 + i * 0.48, 4.5, 0.46, { fontSize: 16 });
      });
      if (tasks.length > 4) d.text(`另 ${tasks.length - 4} 项及完整结果见讲者备注`, 0.5, 6.43, 12.3, 0.2, { fontSize: 10.5, footnote: true, color: C.gray });
    }
  }
  // 14: mechanism-level comparison with an explicit reproducibility boundary.
  {
    const d = slide('结论：按机制选路径，按证据定边界', '保留原项目优化；四种方法独立评估，不把单次结果与跨任务结果混为一谈', ['cache', 'subblock', 'adaln', 'lora', 'overnight']);
    const rows = [['方法', '节省什么', '主要限制'], ['CacheDiT', '命中步的中段计算 / 传输', '残差近似；执行器收益分账'], ['SubBlock', '目标视频 K 块注意力', 'keep 更低，不保证质量同步'], ['AdaLN', '热循环时间投影与规划开销', 'mixed 样本有限；不跳层'], ['LoRA', '减少 DiT 调用次数', '学生模型与采样变化']];
    rows.forEach((row, i) => { const y = 2.01 + i * 0.59; d.text(row[0], 0.5, y, 2.0, 0.45, { fontSize: 18, bold: i === 0 }); d.text(row[1], 2.75, y, 4.7, 0.45, { fontSize: 18, color: i === 0 ? C.teal : C.ink }); d.text(row[2], 7.75, y, 5.05, 0.45, { fontSize: 18, color: i === 0 ? C.teal : C.ink }); });
    d.box('复现链：数据 JSON → 完整 hash / 报告 → 同帧与音画 → 独立验收', 0.5, 5.3, 12.3, 0.65, { fontSize: 20 });
    d.text(short(data.negative_results || '不宣称 Sage 更快；未完成项目继续标为待验收。', 90), 0.5, 6.11, 12.2, 0.45, { fontSize: 18, color: C.orange });
  }
  for (const d of drawings) {
    const { title, sources, extra } = d.note;
    const notes = [title, `数据文件：${path.resolve(dataFile)}`, `快照时间：${data.as_of || '未提供'}；截止：${data.deadline || '未提供'}`,
      '计量：CLI 为该组报告定义的进程端到端时间；DiT 为去噪模型调用计时。外层 hash 与媒体复验不计入 CLI；模型加载内部 hash 按主控口径计入。不是常驻服务稳态。',
      '样本量 n 来自主控 JSON；未提供值保持待测，null 不转换成 0。加速比仅用该方法 base_cli_s / cli_s，不跨组相乘。',
      '质量：SSIM 衡量相对于本组 baseline 的结构相似性，不是画质百分比；音频 RMSE 不等于主观听感。LoRA 改变模型与生成轨迹。',
      'AdaLN：profile 的图/缓冲分配包含执行器分配/等待记账，不能等价归因为逐次 cudaMalloc；正式 mixed 样本与 profile 不混算。',
      data.quality_note || '', data.report_scope || '',
      ...sources.filter(k => REPORTS[k]).map(k => `${SOURCE_NAMES[k]}：${path.join(ROOT, 'docs/research', REPORTS[k])}`),
      `原始来源清单（路径与 SHA）：\n${JSON.stringify(data.sources || {}, null, 2)}`,
      `本页原始数据：\n${JSON.stringify(extra, null, 2)}`,
      `设置：\n${JSON.stringify(data.setup || {}, null, 2)}`,
      `H3_BOUNDS_JSON=${JSON.stringify(d.bounds)}`,
    ].filter(Boolean).join('\n\n');
    d.slide.addNotes(notes);
  }
  return { pptx, drawings };
}
function args(argv) {
  const out = {};
  for (let i = 0; i < argv.length; i++) {
    if (argv[i] === '--help' || argv[i] === '-h') out.help = true;
    else if (['--data', '--output'].includes(argv[i]) && argv[i + 1] && !argv[i + 1].startsWith('--')) out[argv[i].slice(2)] = argv[++i];
    else throw new Error(`Unknown or incomplete argument: ${argv[i]}`);
  }
  return out;
}
async function main() {
  const options = args(process.argv.slice(2));
  if (options.help) { console.log('Usage: node scripts/build_h3_review_deck.js --data DATA.json --output OUT.pptx\nAll figures and charts are editable; PNG comparison frames and local videos remain media.\nOptional AdaLN arrays: samples_s, dit_samples_s, profile_allocation_s, profile_labels.'); return; }
  if (!options.data || !options.output) throw new Error('--data DATA.json and --output OUT.pptx are required');
  const input = path.resolve(options.data), output = path.resolve(options.output);
  if (input === output || path.extname(output).toLowerCase() !== '.pptx') throw new Error('Output must be a separate .pptx path');
  const data = JSON.parse(fs.readFileSync(input, 'utf8'));
  const { pptx, drawings } = buildDeck(data, input);
  fs.mkdirSync(path.dirname(output), { recursive: true });
  await pptx.writeFile({ fileName: output });
  console.log(JSON.stringify({ output, slides: drawings.length, tracked_objects: drawings.reduce((n, d) => n + d.bounds.length, 0), bounds_checked: true, font: FONT, data_as_of: data.as_of || null }));
}
module.exports = { buildDeck, validateData, Drawing, args };
if (require.main === module) main().catch(e => { console.error(`H3 deck: ${e.message}`); process.exitCode = 1; });
