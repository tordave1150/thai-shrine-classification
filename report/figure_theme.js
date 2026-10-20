// Presentation only. Numerical results remain in their original saved units.
const modeColors = { frozen: '#0072B2', partial: '#D55E00', full: '#009E73' };
const symbols = { frozen: 'circle', partial: 'square', full: 'diamond' };
const classColors = ['#7B61A8', '#CC79A7', '#8C6D46', '#5B6770'];
const fontFamily = 'Arial,"Noto Sans Thai","Leelawadee UI",Tahoma,sans-serif';
const ink = '#1F2937', muted = '#6B7280', grid = '#E5E7EB';
const esc = value => String(value).replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;').replaceAll('"', '&quot;');
const styles = 'text{font-family:' + fontFamily + ';fill:' + ink + '}.title{font-size:23px;font-weight:700}.sub{font-size:14px;font-weight:600}.label{font-size:13px}.small{font-size:11px}.axis{stroke-width:1}.grid{stroke:' + grid + ';stroke-width:1}';
function heatColor(percent) {
  const t = Math.max(0, Math.min(100, percent)) / 100;
  const low = [244, 247, 248], high = [14, 74, 92];
  return 'rgb(' + low.map((v, i) => Math.round(v + t * (high[i] - v))).join(',') + ')';
}
module.exports = { modeColors, symbols, classColors, fontFamily, ink, muted, grid, styles, heatColor, esc };
