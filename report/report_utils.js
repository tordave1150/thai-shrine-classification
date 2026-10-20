const fs = require('fs');
const path = require('path');
const crypto = require('crypto');
const root = path.resolve(__dirname, '..');
function parseCsv(text) {
  const rows = []; let row = [], field = '', quoted = false;
  text = text.replace(/^\uFEFF/, '');
  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    if (quoted) {
      if (c === '"' && text[i + 1] === '"') { field += '"'; i++; }
      else if (c === '"') quoted = false;
      else field += c;
    } else if (c === '"') quoted = true;
    else if (c === ',') { row.push(field); field = ''; }
    else if (c === '\n') { row.push(field.replace(/\r$/, '')); rows.push(row); row = []; field = ''; }
    else field += c;
  }
  if (quoted) throw Error('Unclosed CSV quote');
  if (field || row.length) { row.push(field.replace(/\r$/, '')); rows.push(row); }
  const header = rows.shift();
  return rows.filter(r => r.some(Boolean)).map(r => {
    if (r.length !== header.length) throw Error('CSV column mismatch');
    return Object.fromEntries(header.map((key, i) => [key, r[i]]));
  });
}
const readCsv = relative => parseCsv(fs.readFileSync(path.join(root, relative), 'utf8'));
const hash = relative => crypto.createHash('sha256').update(fs.readFileSync(path.join(root, relative))).digest('hex');
function sharpRuntime() {
  try { return require('sharp'); }
  catch { return require(path.join(process.env.USERPROFILE, '.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/sharp')); }
}
const mean = values => values.reduce((a, b) => a + b, 0) / values.length;
const sd = values => { const m = mean(values); return Math.sqrt(values.reduce((a, b) => a + (b - m) ** 2, 0) / (values.length - 1)); };
module.exports = { root, parseCsv, readCsv, hash, sharpRuntime, mean, sd };
