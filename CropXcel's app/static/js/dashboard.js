/* === Dashboard JS (moved from <script> in dashboard.html) === */

// 0) Bootstrap: read Django data (injected as JSON in the template)
(function(){
  const el = document.getElementById('dashboard-data');
  if (!el) return;
  try {
    window.CropXcel = JSON.parse(el.textContent);
  } catch(e) {
    console.error('Failed to parse dashboard-data JSON', e);
    window.CropXcel = {};
  }
})();

const {
  JOB_ID, BOUNDS, OVERLAY_URL, HOTSPOTS,
  PROBE_BIN, PROBE_META, FIELD_ID
} = window.CropXcel || {};

console.log("%cCropXcel dashboard.js v3 (emoji+why)","background:#111;color:#0ff;padding:2px 6px;border-radius:6px");
// ============== MAP ==================
const map = L.map('map');

L.tileLayer(
  'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
  { attribution: 'Tiles © Esri' }
).addTo(map);

map.fitBounds(BOUNDS);
map.createPane('overlayPane');
map.getPane('overlayPane').style.zIndex = 450;

// Keep a reference so we can show/hide it later
let overlayLayer = null;
let overlayVisible = true;

if (OVERLAY_URL && OVERLAY_URL.endsWith('.png')) {
  overlayLayer = L.imageOverlay(OVERLAY_URL, BOUNDS, { opacity:0.75, pane:'overlayPane' }).addTo(map);
}

function setOverlayVisible(on){
  overlayVisible = !!on;
  if (!overlayLayer) return;
  overlayLayer.setOpacity(overlayVisible ? 0.75 : 0.0);
  showToast(`Overlay: ${overlayVisible ? 'ON' : 'OFF'}`);
}

// Right-click toggles overlay
map.on('contextmenu', ()=>{
  if (!overlayLayer) return;
  setOverlayVisible(!overlayVisible);
});

// Press-and-hold Shift → temporarily hide overlay
let shiftHeld = false;
document.addEventListener('keydown', (e)=>{
  if (e.key === 'Shift' && !shiftHeld){
    shiftHeld = true;
    if (overlayLayer && overlayVisible){ overlayLayer.setOpacity(0.0); showToast('Overlay: OFF (Shift)'); }
  }
});
document.addEventListener('keyup', (e)=>{
  if (e.key === 'Shift'){
    shiftHeld = false;
    if (overlayLayer && overlayVisible){ overlayLayer.setOpacity(0.75); showToast('Overlay: ON'); }
  }
});

// Fallback mapping (normalized)
const REASON_FALLBACK = {
  "radar drop (water)": { emoji: "💧", explain: "Dark radar patch = standing water in field" },
  "vh/vv ratio change": { emoji: "🌱", explain: "Soil/water signal shift = possible saturation" },
  "signal variability": { emoji: "⚡", explain: "Unstable radar = uneven wet/dry spots" },
  "water detected": { emoji: "💧", explain: "Standing water detected in field" },
  "high moisture": { emoji: "🌊", explain: "Soil moisture levels are elevated" },
  "flooding risk": { emoji: "🚨", explain: "High risk of waterlogging detected" },
  "drainage needed": { emoji: "🚰", explain: "Field drainage may be required" },
  "unknown": { emoji: "❓", explain: "Reason for risk is unclear" }
};
const norm = s => String(s||"").toLowerCase().replace(/\s+/g," ").trim();
function withFriendly(reason, emoji, explain){
  console.log("withFriendly called with:", {reason, emoji, explain}); // Debug log
  const normalizedReason = norm(reason);
  const f = REASON_FALLBACK[normalizedReason];
  console.log("Normalized reason:", normalizedReason, "Found fallback:", f); // Debug log
  
  const result = { 
    emoji: emoji || (f ? f.emoji : "ℹ️"), 
    explain: explain || (f ? f.explain : "") 
  };
  console.log("withFriendly result:", result); // Debug log
  return result;
}


// ===== Helpers (toast + risk utils)
const pill = document.getElementById("hoverpill");
let toastTimer = null;
function showToast(msg){
  const el = pill;
  el.style.display='block';
  el.style.left='12px';
  el.style.top='12px';
  el.innerHTML = `<b>${msg}</b>`;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(()=>{ el.style.display='none'; }, 3000);
}
const clamp01 = (x)=> Math.max(0, Math.min(1, x));
const pct = (x)=> Math.round(clamp01(x)*100);
const levelFrom  = (v)=> v>=0.7 ? "High" : v>=0.4 ? "Caution" : "Low";
const colorFor   = (v)=> v>=0.7 ? '#e31a1c' : v>=0.4 ? '#f59e0b' : '#2ecc71';
function tipsFor(level){
  if (level==="High")    return "Standing water likely. Drain within 24h; open outlets/pump.";
  if (level==="Caution") return "Watch next 2–3 days. Avoid heavy irrigation; check after rain.";
  return "Normal. Maintain current irrigation schedule.";
}

// ===== Hotspots layer (same logic as before)
function colorFromPercent(p){
  if (p >= 70) return '#ef4444';      // Alert
  if (p >= 40) return '#f59e0b';      // Watch
  return '#22c55e';                   // Healthy/Low
}

if (HOTSPOTS) {
  fetch(HOTSPOTS, { cache: 'no-store' })
    .then(r => r.json())
    .then(gj => {
      console.log("Hotspots GeoJSON loaded:", gj); // Debug log
      
      // Log the first feature to see its structure
      if (gj.features && gj.features.length > 0) {
        console.log("First hotspot feature:", gj.features[0]);
        console.log("Properties of first hotspot:", gj.features[0].properties);
      }
      
      const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));

      const hotspotLayer = L.geoJSON(gj, {
        pointToLayer: (f, latlng) => {
          const riskRaw = f.properties?.risk ?? f.properties?.risk_pct ?? 0;
          const riskPercent = clamp((riskRaw <= 1 ? Math.round(riskRaw * 100) : Math.round(riskRaw)), 0, 100);
          const level   = f.properties?.level || (riskPercent >= 70 ? 'Alert' : (riskPercent >= 40 ? 'Watch' : 'Healthy'));
          const areaHa  = f.properties?.area_ha;
          const reason  = f.properties?.reason || 'Unknown risk detected';
          const action  = f.properties?.action || '';
          
          // Debug log the properties
          console.log("Hotspot properties:", {
            reason: f.properties?.reason,
            reason_emoji: f.properties?.reason_emoji,
            reason_explain: f.properties?.reason_explain
          });
          
          const friendly = withFriendly(
            reason,
            f.properties?.reason_emoji,
            f.properties?.reason_explain
          );
          const reasonEmoji   = friendly.emoji;
          const reasonExplain = friendly.explain;
          const color   = colorFromPercent(riskPercent);
          const chartB64 = f.properties?.chart_b64;

          const uidInfo  = 'info_'  + Math.random().toString(36).slice(2, 8);
          const uidChart = 'chart_' + Math.random().toString(36).slice(2, 8);
          const uidBar   = 'bar_'   + Math.random().toString(36).slice(2, 8);
          const uidRiskRow = 'risk_' + Math.random().toString(36).slice(2, 8);
          const uidBtn   = 'btn_'   + Math.random().toString(36).slice(2, 8);

          const html = `
            <div style="min-width:320px;max-width:400px;background:#fff;border-radius:16px;
                        box-shadow:0 10px 30px rgba(2,6,23,.20);padding:18px 20px;
                        font-family:system-ui,Segoe UI,Roboto,Arial,sans-serif;color:#0f172a;line-height:1.55">

              <div style="display:flex;align-items:center;gap:10px;margin-bottom:12px">
                <div style="width:14px;height:14px;border-radius:50%;background:${color};
                            box-shadow:0 0 0 2px #fff, 0 0 0 6px ${color}22"></div>
                <div style="font-weight:800;font-size:15px">Hotspot detected</div>
              </div>

              <div id="${uidRiskRow}" style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px">
                <div style="font-size:14px"><strong>Water Risk</strong></div>
                <div style="font-size:14px"><strong>${riskPercent}% · ${level}</strong></div>
              </div>

              <div id="${uidBar}" style="position:relative;height:12px;border-radius:999px;overflow:hidden;margin-bottom:14px;background:#e5e7eb">
                <div style="position:absolute;left:0;top:0;bottom:0;width:${riskPercent}%;background:${color};border-radius:999px"></div>
              </div>

              <div id="${uidInfo}">
                ${areaHa!=null ? `
                <div style="margin-bottom:12px">
                  <div style="font-size:11px;letter-spacing:.04em;text-transform:uppercase;color:#64748b;font-weight:700">Size</div>
                  <div style="font-size:13px;color:#0f172a;margin-top:2px">${Number(areaHa).toFixed(2)} ha</div>
                </div>` : ``}

                <div style="margin-bottom:12px">
                  <div style="font-size:11px;letter-spacing:.04em;text-transform:uppercase;color:#64748b;font-weight:700">Why risky</div>
                  <div style="display:flex;gap:8px;align-items:flex-start;margin-top:4px">
                    <div style="font-size:18px;line-height:1">${reasonEmoji || "⚠️"}</div>
                    <div>
                      <div style="font-size:13px;color:#0f172a;"><b>${reason || "Risk detected"}</b></div>
                      ${reasonExplain ? `<div style="font-size:12px;color:#475569;margin-top:2px">${reasonExplain}</div>` : `<div style="font-size:12px;color:#475569;margin-top:2px">Risk analysis indicates potential waterlogging</div>`}
                    </div>
                  </div>
                </div>

                ${action ? `
                <div style="margin-bottom:2px">
                  <div style="font-size:11px;letter-spacing:.04em;text-transform:uppercase;color:#64748b;font-weight:700">What you should do</div>
                  <div style="font-size:13px;color:#0f172a;margin-top:2px">${action}</div>
                </div>` : ``}
              </div>

              ${chartB64 ? `
              <div id="${uidChart}" style="display:none">
                <div style="font-size:11px;letter-spacing:.04em;text-transform:uppercase;color:#64748b;font-weight:700;margin-bottom:8px">Why risky</div>
                <img src="data:image/png;base64,${chartB64}" alt="Why risky contributions"
                    style="display:block;width:100%;height:auto;border-radius:8px;image-rendering:-webkit-optimize-contrast" />
              </div>` : ``}

              ${chartB64 ? `
              <button id="${uidBtn}" aria-pressed="false"
                onclick="(function(btn){
                  const info  = document.getElementById('${uidInfo}');
                  const chart = document.getElementById('${uidChart}');
                  const bar   = document.getElementById('${uidBar}');
                  const risk  = document.getElementById('${uidRiskRow}');
                  const showingChart = chart.style.display !== 'none';
                  if (showingChart){
                    chart.style.display='none'; info.style.display='block';
                    if(bar)  bar.style.display='block';
                    if(risk) risk.style.display='flex';
                    btn.textContent='Show why risky'; btn.setAttribute('aria-pressed','false');
                  } else {
                    info.style.display='none'; chart.style.display='block';
                    if(bar)  bar.style.display='none';
                    if(risk) risk.style.display='none';
                    btn.textContent='Back to info'; btn.setAttribute('aria-pressed','true');
                  }
                })(this)"
                style="display:block;width:100%;padding:10px 0;border:none;border-radius:999px;
                      background:#3b82f6;color:#fff;font-size:13px;font-weight:700;
                      box-shadow:0 1px 0 rgba(255,255,255,.4) inset, 0 2px 10px rgba(59,130,246,.35);
                      cursor:pointer;margin-top:12px;transition:transform .06s ease, background .2s ease;">
                Show why risky
              </button>` : ``}
            </div>`;

          const m = L.circleMarker(latlng, {
            radius: 6, color: '#0f172a', weight: 1, fillColor: color, fillOpacity: 0.9
          });

          m.bindPopup(html, { maxWidth: 440, className: 'hotspot-popup' });
          return m;
        }
      });

      hotspotLayer.addTo(map);
    })
    .catch(()=>{});
}

// ===== Client-side probe (unchanged logic, just moved) =====
let probeMeta=null, probeArray=null, maskArray=null, dtype=null, canClient=false;
let EFFECTIVE_BOUNDS = BOUNDS;

(async function(){
  try{
    if (!PROBE_BIN || !PROBE_META) return;
    probeMeta = await fetch(PROBE_META, {cache:'no-store'}).then(r=>r.json());
    const W = probeMeta.cols ?? probeMeta.width;
    const H = probeMeta.rows ?? probeMeta.height;
    probeMeta.width = Number(W); probeMeta.height = Number(H);
    probeMeta.scale = probeMeta.scale || 1000;
    if (Array.isArray(probeMeta.web_bounds)) EFFECTIVE_BOUNDS = probeMeta.web_bounds;

    const buf = await fetch(PROBE_BIN, {cache:'no-store'}).then(r=>r.arrayBuffer());
    const npx = probeMeta.width * probeMeta.height;
    const layout = probeMeta.layout || {};
    const dataBytes = layout.data_bytes ?? (npx*2);
    const maskBytes = layout.mask_bytes ?? 0;

    if (buf.byteLength < dataBytes) throw new Error("Probe buffer smaller than data_bytes");
    const dataView = buf.slice(0, dataBytes);
    const maskView = (maskBytes>0 && buf.byteLength >= dataBytes+maskBytes) ? buf.slice(dataBytes, dataBytes+maskBytes) : null;

    if (dataBytes === npx)       { probeArray = new Uint8Array(dataView);  dtype='u8'; }
    else if (dataBytes === npx*2){ probeArray = new Uint16Array(dataView); dtype='u16'; }
    else if (dataBytes === npx*4){ probeArray = new Float32Array(dataView);dtype='f32'; }
    else {
      probeArray = (dataBytes % 4 === 0) ? new Float32Array(dataView)
                : (dataBytes % 2 === 0) ? new Uint16Array(dataView)
                : new Uint8Array(dataView);
      dtype = (probeArray.BYTES_PER_ELEMENT===4) ? 'f32' : (probeArray.BYTES_PER_ELEMENT===2 ? 'u16' : 'u8');
      console.warn("Nonstandard data_bytes; inferred dtype =", dtype);
    }

    if (maskView) {
      maskArray = new Uint8Array(maskView);
      if (maskArray.length !== npx) { console.warn("Mask size mismatch (ignoring mask)", maskArray.length, "vs", npx); maskArray = null; }
    }
    canClient = !!(probeArray && probeMeta.width && probeMeta.height);
    console.log("Probe ready:", {dtype, w:probeMeta.width, h:probeMeta.height, scale:probeMeta.scale, hasMask:!!maskArray});
  }catch(e){
    console.warn("Probe load failed:", e);
    canClient=false;
  }
})();

function sampleClient(latlng){
  if (!canClient) return null;
  const [[S,W],[N,E]] = EFFECTIVE_BOUNDS;
  const x = Math.floor(((latlng.lng - W) / (E - W)) * probeMeta.width);
  const y = Math.floor(((N - latlng.lat) / (N - S)) * probeMeta.height);
  if (x<0 || x>=probeMeta.width || y<0 || y>=probeMeta.height) return null;
  const idx = y*probeMeta.width + x;
  if (maskArray && maskArray[idx] === 0) return null;

  let v01;
  if (dtype==='u8')       v01 = probeArray[idx] / 255;
  else if (dtype==='u16') v01 = probeArray[idx] / (probeMeta.scale || 1000);
  else {
    const raw = probeArray[idx];
    const hasMM = Number.isFinite(probeMeta.min) && Number.isFinite(probeMeta.max) && (probeMeta.max>probeMeta.min);
    v01 = hasMM ? (raw - probeMeta.min) / (probeMeta.max - probeMeta.min) : raw;
  }
  if (!Number.isFinite(v01)) return null;
  return { v: clamp01(v01) };
}

// Smooth hover pill + server fallback
let lastLL=null, emaV=null, raf=0, srvCooldown=false;
function renderPill(ll){
  const c = sampleClient(ll);
  if (!c){ pill.style.display = 'none'; emaV = null; return; }
  emaV = (emaV==null) ? c.v : (0.6*emaV + 0.4*c.v);
  const v = clamp01(emaV);
  const p = pct(v), lvl = levelFrom(v), col = colorFor(v);

  const pt = map.latLngToContainerPoint(ll);
  pill.style.display='block'; pill.style.left=(pt.x+14)+'px'; pill.style.top=(pt.y+14)+'px';
  pill.innerHTML = `
    <div style="display:flex;flex-direction:column;gap:2px;min-width:180px">
      <div>Waterlogging risk · ${p}% · ${lvl}</div>
      <div style="width:100%;height:6px;background:#e5e7eb;border-radius:999px;overflow:hidden">
        <span style="display:block;height:100%;width:${p}%;background:${col}"></span>
      </div>
    </div>`;
}
function loop(){
  raf = requestAnimationFrame(loop);
  if (lastLL) renderPill(lastLL);
  if (!canClient && lastLL && !srvCooldown){
    srvCooldown = true; setTimeout(()=>srvCooldown=false, 200);
    fetch(`/probe/${JOB_ID}?lat=${lastLL.lat}&lon=${lastLL.lng}`)
      .then(r=>r.ok?r.json():null).then(d=>{
        if (d && typeof d.value==='number'){
          const v = clamp01(d.value);
          emaV = (emaV==null) ? v : (0.6*emaV + 0.4*v);
        }
      }).catch(()=>{});
  }
}
map.on('mousemove', e=>{ lastLL=e.latlng; if(!raf) loop(); });
map.on('mouseout', ()=>{ lastLL=null; emaV=null; pill.style.display='none'; cancelAnimationFrame(raf); raf=0; });

map.on('click', async (e)=>{
  let v = sampleClient(e.latlng)?.v;
  if (v==null && !canClient){
    try{
      const r = await fetch(`/probe/${JOB_ID}?lat=${e.latlng.lat}&lon=${e.latlng.lng}`);
      const d = await r.json(); if (typeof d?.value==='number') v = clamp01(d.value);
    }catch{}
  }
  if (v==null) return;
  const p=pct(v), lvl=levelFrom(v), col=colorFor(v);
  const html = `
    <div style="min-width:260px">
      <b>Waterlogging risk</b>
      <div style="margin-top:6px;font-size:12px;color:#334155">Value: ${p}% · <b>${lvl}</b></div>
      <div style="width:100%;height:8px;background:#e5e7eb;border-radius:999px;margin-top:6px;overflow:hidden">
        <span style="display:block;height:100%;width:${p}%;background:${col}"></span>
      </div>
      <div style="margin-top:8px;font-size:12px;color:#64748b"><i>Tip:</i> ${tipsFor(lvl)}</div>
    </div>`;
  L.popup({offset:[0,-6]}).setLatLng(e.latlng).setContent(html).openOn(map);
});

// ============== Insights + Scale ==============
const tableWrap = document.getElementById('insightsTable');
const title     = document.getElementById('insTitle');
const moreBtn   = document.getElementById('toggleInsights');
const techBtn   = document.getElementById('techBtn');
const exitBtn   = document.getElementById('exitMapOnlyBtn');
const legendDiv = document.getElementById('legendRows');
const donutCtr  = document.getElementById('donutCenter');

let farmerHTML = "", techHTML = "", scaleHTML = "", plotHTML = "";
let techMode = false, expandedFarmer=false, expandedTech=false;

function applyRowLimit(t, max){
  if (!t || !t.tBodies[0]) return;
  const rows = Array.from(t.tBodies[0].rows);
  
  console.log("🚨 applyRowLimit called with max:", max);
  console.log("🚨 Total rows:", rows.length);
  
  rows.forEach((r,i)=>{
    // Look for status span with class names
    const statusSpan = r.querySelector('span.healthy, span.watch, span.alert');
    let status = 'unknown';
    if (statusSpan) {
      if (statusSpan.classList.contains('healthy')) status = 'Healthy';
      else if (statusSpan.classList.contains('watch')) status = 'Watch';
      else if (statusSpan.classList.contains('alert')) status = 'Alert';
    }
    
    const willShow = i < max;
    
    console.log(`🚨 Row ${i}: status="${status}", willShow=${willShow}`);
    
    // Explicitly set display to table-row for visible rows
    r.style.display = willShow ? 'table-row' : 'none';
    
    if (!willShow && status === 'Alert') {
      console.log("🔥 HIDING ALERT ROW!");
    }
  });
  
  console.log("✅ applyRowLimit complete. Visible rows:", max);
}
function sortByColumn(table, colIdx, thEl, dir){
  const tbody = table.tBodies[0];
  const rows  = Array.from(tbody.rows);
  const asc = (dir === 'asc') ? true : (dir === 'desc') ? false : !thEl.classList.contains('asc');
  const parseVal = (s)=>{
    const txt = (s||'').trim();
    if (/^\d{4}-\d{2}-\d{2}$/.test(txt)) return new Date(txt).getTime();
    const n = parseFloat(txt.replace(/[, ]/g,'')); return Number.isFinite(n) ? n : txt;
  };
  rows.sort((a,b)=>{
    const A = parseVal(a.cells[colIdx]?.innerText);
    const B = parseVal(b.cells[colIdx]?.innerText);
    if (typeof A === 'number' && typeof B === 'number') return asc ? (A-B) : (B-A);
    return asc ? String(A).localeCompare(String(B)) : String(A).localeCompare(String(B)) * -1;
  });
  table.querySelectorAll('th').forEach(th=>th.classList.remove('asc'));
  if (asc) thEl.classList.add('asc');
  rows.forEach(r=>tbody.appendChild(r));
}
function makeSortableAndSparky(t, isTech){
  if (!t) return;
  const ths = t.querySelectorAll('thead th');
  ths.forEach((th, idx)=>{ th.classList.add('sortable'); th.onclick = () => sortByColumn(t, idx, th); });
  if (!isTech) return;
  const rows = Array.from(t.tBodies[0]?.rows || []);
  const cols = ths.length;
  const isNumeric = Array(cols).fill(false);
  for (let c=1; c<cols; c++){
    const sample = rows.slice(0, Math.min(12, rows.length))
      .map(r => parseFloat((r.cells[c]?.innerText || '').replace(/[, ]/g,'')));
    const valid = sample.filter(v => Number.isFinite(v)).length;
    isNumeric[c] = valid >= Math.ceil(sample.length*0.6);
  }
  const min = Array(cols).fill(+Infinity), max = Array(cols).fill(-Infinity);
  rows.forEach(r=>{
    for (let c=1; c<cols; c++){
      if (!isNumeric[c]) continue;
      const v = parseFloat((r.cells[c].innerText||'').replace(/[, ]/g,''));
      if (Number.isFinite(v)){ if (v<min[c]) min[c]=v; if (v>max[c]) max[c]=v; }
    }
  });
  rows.forEach(r=>{
    for (let c=1; c<cols; c++){
      if (!isNumeric[c]) continue;
      const cell = r.cells[c];
      const v = parseFloat((cell.innerText||'').replace(/[, ]/g,''));
      let p = 0; if (Number.isFinite(v) && max[c] > min[c]) p = (v-min[c])/(max[c]-min[c])*100;
      cell.classList.add('spark'); cell.style.setProperty('--pct', p.toFixed(2));
    }
  });
}
function ensureFarmerColgroup(t) {
  if (!t || t.classList.contains('has-colgroup')) return;
  if (!t.querySelector('colgroup')) {
    const cg = document.createElement('colgroup');
    ['90px','110px','auto'].forEach(w=>{
      const c = document.createElement('col');
      if (w !== 'auto') c.style.width = w;
      cg.appendChild(c);
    });
    t.insertBefore(cg, t.firstChild);
  }
  t.classList.add('has-colgroup');
}
// Build a lightweight preview (date, status, advice) from the first few rows of the farmer table
function generateInsightPreview(table, limit=3){
  const previewEl = document.getElementById('insightsPreview');
  if (!previewEl || !table || !table.tBodies || !table.tBodies[0]) return;

  const all = Array.from(table.tBodies[0].rows || []);
  const rows = all.slice(0, Math.max(1, limit));
  if (!rows.length){ previewEl.innerHTML = ''; return; }

  const toCell = (row, sel, idx) => row.querySelector(sel) || row.cells[idx] || null;
  let html = '<div class="insight-preview-grid">';
  rows.forEach(r=>{
    const dateCell   = toCell(r, 'td.date-cell',   0);
    const statusCell = toCell(r, 'td.status-cell', 1);
    const actionCell = toCell(r, 'td.action-cell', 2);
    const date   = (dateCell?.innerText || '').trim();
    const sSpan  = statusCell ? statusCell.querySelector('span') : null;
    const status = (sSpan?.innerText || statusCell?.innerText || '').trim();
    const sClass = sSpan?.className || '';
    const advice = (actionCell?.innerText || '').trim();
    html += `
      <div class="insight-preview-item">
        <div class="preview-date">${date}</div>
        <div class="preview-status"><span class="${sClass}">${status}</span></div>
        <div class="preview-advice">${advice}</div>
      </div>`;
  });
  html += '</div>';
  previewEl.innerHTML = html;
}
function buildDonutMatchBars(){
  const svg  = document.querySelector('.scale-donut');
  const segs = svg ? svg.querySelectorAll('.segments .seg') : null;
  const rows = Array.from(document.querySelectorAll('.legend .legrow'));
  if (!svg || !segs || !rows.length) return;

  const totalHa = rows.reduce((s, r) => s + (parseFloat(r.dataset.ha) || 0), 0);
  const val = (totalHa || 0);
  const valStr = (val < 10) ? val.toFixed(2) : (val < 100 ? val.toFixed(1) : Math.round(val).toString());
  const center = document.getElementById('donutCenter');
  if (center) center.textContent = valStr;

  segs.forEach((seg, i) => {
    const row = rows[i]; if (!row) return;
    const pct = Math.max(0, Math.min(100, parseFloat(row.dataset.pct)||0));
    const color = row.style.getPropertyValue('--c') || getComputedStyle(row).getPropertyValue('--c') || '#9ca3af';
    seg.setAttribute('stroke', color.trim());
    seg.setAttribute('stroke-dasharray', `${pct} ${100 - pct}`);
    seg.setAttribute('stroke-dashoffset', '0');
  });
}
function renderTables(){
  console.log("🔧 renderTables() called at:", new Date().toISOString());
  console.log("🔧 Tech mode:", window.__techMode__);
  console.log("🔧 Expanded Farmer:", window.__expandedFarmer__);
  console.log("🔧 Expanded Tech:", window.__expandedTech__);
  console.log("🔧 Farmer HTML length:", window.__farmerHTML__?.length || 0);
  console.log("🔧 Tech HTML length:", window.__techHTML__?.length || 0);
  
  const maxF = Number(tableWrap?.dataset?.max ?? 5);
  const maxT = Number(tableWrap?.dataset?.techMax ?? 5);
  
  console.log("🔧 maxF (farmer rows to show):", maxF);
  console.log("🔧 maxT (tech rows to show):", maxT);

  if (window.__techMode__) {
    title.innerHTML = 'Technical Details <span class="hint" data-tip="Detailed raw indicators from Sentinel-1 (VH/VV, ratios, etc.). Useful for agronomists and advanced users."></span>';
    tableWrap.innerHTML = window.__techHTML__ || "<div class='empty'>No records.</div>";
    moreBtn.innerHTML = window.__expandedTech__ ? '<i class="fas fa-chevron-up"></i> Show less' : '<i class="fas fa-chevron-down"></i> Show more';
    techBtn.textContent = 'Exit Technical Details';
    const previewEl = document.getElementById('insightsPreview'); if (previewEl) previewEl.style.display='none';
    const t = tableWrap.querySelector('table');
    if (t){ t.classList.add('minitable','tech'); makeSortableAndSparky(t, true);
      const dateTH = t.querySelector('thead th:first-child'); if (dateTH) sortByColumn(t, 0, dateTH, 'desc');
      if (!window.__expandedTech__) applyRowLimit(t, maxT);
      
      // Update button with row count
      const totalRows = t.tBodies[0]?.rows.length || 0;
      const hiddenRows = Math.max(0, totalRows - maxT);
      if (!window.__expandedTech__ && hiddenRows > 0) {
        moreBtn.innerHTML = `<i class="fas fa-chevron-down"></i> Show ${hiddenRows} more`;
      }
    }
  } else {
    title.innerHTML = 'Per-pass Insights <span class="hint" data-tip="Shows each satellite pass. Farmers can check the date, risk status, and suggested action. Helpful for deciding when to irrigate or drain."></span>';
    
    // Debug what we're setting
    console.log("🔧 Setting farmer HTML (first 200 chars):", window.__farmerHTML__?.substring(0, 200));
    
    console.log("🔧 About to set tableWrap.innerHTML");
    console.log("🔧 Farmer HTML length:", window.__farmerHTML__?.length);
    console.log("🔧 Farmer HTML sample:", window.__farmerHTML__?.substring(0, 300));
    
    tableWrap.innerHTML = window.__farmerHTML__ || "<div class='empty'>No records.</div>";
    
    // Make sure table wrapper is visible
    tableWrap.style.display = 'block';
    tableWrap.style.visibility = 'visible';
    tableWrap.style.opacity = '1';
    
    console.log("🔧 tableWrap.innerHTML after setting:", tableWrap.innerHTML.substring(0, 300));
    
    moreBtn.innerHTML = window.__expandedFarmer__ ? '<i class="fas fa-chevron-up"></i> Show less' : '<i class="fas fa-chevron-down"></i> Show more';
    techBtn.textContent = 'Show Technical Details';

  const t = tableWrap.querySelector('table');
    console.log("🔧 Found table:", !!t);
    if (t){
      console.log("🔧 Table found, processing...");
      ensureFarmerColgroup(t);
      t.classList.add('minitable','farmer');
      const dateTH = t.querySelector('thead th:first-child');
      if (dateTH) sortByColumn(t, 0, dateTH, 'desc');
      
      const totalRows = t.tBodies[0]?.rows.length || 0;
      console.log("🔧 Total rows in table:", totalRows);
      
      if (!window.__expandedFarmer__) {
        console.log("🔧 Applying row limit:", maxF);
        applyRowLimit(t, maxF);
      }
      // Build and show preview when not expanded (only if preview container exists)
      const previewEl = document.getElementById('insightsPreview');
      if (!window.__expandedFarmer__) {
        if (previewEl) {
          previewEl.style.display='block';
          generateInsightPreview(t, maxF);
          // Hide full table only when preview is present
          tableWrap.style.display = 'none';
        } else {
          // No preview container; keep the limited table visible
          tableWrap.style.display = 'block';
        }
      } else {
        if (previewEl) previewEl.style.display='none';
        tableWrap.style.display = 'block';
      }
      
      // Update button with row count
      const hiddenRows = Math.max(0, totalRows - maxF);
      console.log("🔧 Hidden rows:", hiddenRows);
      if (!window.__expandedFarmer__ && hiddenRows > 0) {
        moreBtn.innerHTML = `<i class="fas fa-chevron-down"></i> Show ${hiddenRows} more`;
      }
    } else {
      console.log("⚠️ No table found in tableWrap!");
      console.log("⚠️ tableWrap.innerHTML:", tableWrap?.innerHTML?.substring(0, 500));
      const previewEl = document.getElementById('insightsPreview'); if (previewEl) previewEl.style.display='none';
    }
  }
  
  // Update button expanded state
  moreBtn.classList.toggle('expanded', window.__techMode__ ? window.__expandedTech__ : window.__expandedFarmer__);
  
  buildDonutMatchBars();
}

// UI state toggles
window.__techMode__ = false;
window.__expandedFarmer__ = false;
window.__expandedTech__   = false;

moreBtn.onclick = ()=>{
  if (window.__techMode__) window.__expandedTech__ = !window.__expandedTech__;
  else                     window.__expandedFarmer__ = !window.__expandedFarmer__;
  renderTables();
};
techBtn.onclick = ()=>{ window.__techMode__ = !window.__techMode__; renderTables(); };

function setMapOnly(on){
  document.body.classList.toggle('map-only', !!on);
  const isOn = document.body.classList.contains('map-only');
  
  // Show helpful toast messages
  if (isOn) {
    showToast('🗺️ Map only mode • Use the Exit button or press Escape to exit');
  } else {
    showToast('📊 Back to full dashboard');
  }
  
  // Invalidate map size after layout changes with multiple attempts
  setTimeout(() => {
    if (typeof map !== 'undefined' && map.invalidateSize) {
      map.invalidateSize(true);
    }
  }, 100);
  
  setTimeout(() => {
    if (typeof map !== 'undefined' && map.invalidateSize) {
      map.invalidateSize(true);
    }
  }, 300);
}
if (exitBtn) {
  exitBtn.onclick = () => {
    setMapOnly(false);
    
    // Update fullscreen button to reflect normal mode
    const fullscreenBtn = document.getElementById('fullscreenBtn');
    if (fullscreenBtn) {
      const icon = fullscreenBtn.querySelector('i');
      if (icon) {
        icon.className = 'fas fa-expand';
        fullscreenBtn.title = 'Enter Fullscreen Mode (F)';
        fullscreenBtn.setAttribute('aria-label', 'Enter fullscreen mode');
      }
    }
    showToast('📍 Returned to dashboard view');
  };
}

// Keyboard shortcuts
document.addEventListener('keydown', (e) => {
  // Escape to exit map-only mode
  if (e.key === 'Escape') {
    const isCurrentlyMapOnly = document.body.classList.contains('map-only');
    if (isCurrentlyMapOnly) {
      setMapOnly(false);
      
      // Update fullscreen button to reflect normal mode
      const fullscreenBtn = document.getElementById('fullscreenBtn');
      if (fullscreenBtn) {
        const icon = fullscreenBtn.querySelector('i');
        if (icon) {
          icon.className = 'fas fa-expand';
          fullscreenBtn.title = 'Enter Fullscreen Mode (F)';
          fullscreenBtn.setAttribute('aria-label', 'Enter fullscreen mode');
        }
      }
      showToast('📍 Exited fullscreen mode');
    }
  }
  
  // 'F' key to toggle fullscreen (when not in an input field)
  if (e.key === 'f' || e.key === 'F') {
    if (!e.target.matches('input, textarea, [contenteditable]')) {
      e.preventDefault();
      const fullscreenBtn = document.getElementById('fullscreenBtn');
      if (fullscreenBtn && !fullscreenBtn.disabled) {
        fullscreenBtn.click();
      }
    }
  }
});

// Function to load insights data
async function loadInsights() {
  console.log("🚨 loadInsights() called - about to override table data!");
  
  try{
    // Use multiple cache-busting techniques
    const timestamp = Date.now();
    const jobId = window.CropXcel?.JOB_ID || '1';
    const res = await fetch(`/fields/${FIELD_ID}/insights/?t=${timestamp}&v=${jobId}&_=${Math.random()}`, {
      cache: 'no-store',
      headers: {
        'Cache-Control': 'no-cache, no-store, must-revalidate',
        'Pragma': 'no-cache',
        'Expires': '0'
      }
    });
    const data = await res.json();
    
    console.log("🚨 Received AJAX data:", {
      farmer_html_length: data.farmer_table_html?.length || 0,
      farmer_html_preview: data.farmer_table_html?.substring(0, 300)
    });
    
    window.__farmerHTML__ = data.farmer_table_html || "";
    window.__techHTML__   = data.technical_table_html || "";
    const scaleHTML       = data.legend_rows_html || "";
    const plotHTML        = data.plot_section || "";
    document.getElementById("plotSection").innerHTML = plotHTML;
    document.getElementById("legendRows").innerHTML  = scaleHTML;
    
    // Hide skeletons and show content
    const insightsSkeleton = document.getElementById("insightsSkeleton");
    if (insightsSkeleton) insightsSkeleton.style.display = 'none';
    if (tableWrap) {
      tableWrap.style.display = 'block';
      console.log("✅ Table wrapper display set to block");
    }
    
    console.log("🚨 About to call renderTables() - this will replace the table!");
    console.log("🚨 Farmer HTML exists:", !!window.__farmerHTML__);
    console.log("🚨 Farmer HTML length:", window.__farmerHTML__?.length);
    renderTables();
    
  }catch(err){
    console.error("Insights load failed:", err);
    window.__farmerHTML__ = ""; window.__techHTML__ = "";
    document.getElementById("plotSection").innerHTML = "";
    document.getElementById("legendRows").innerHTML  = "";
    renderTables();
  }
}

// Load insights chunks generated by your view
loadInsights(); // ✅ Re-enabled to test row limit theory
console.log("🔧 loadInsights() re-enabled for testing");

// ============== NEW UI ENHANCEMENTS ==================

// Animated counter for stat values
function animateValue(element, start, end, duration) {
  if (!element) return;
  const range = end - start;
  const increment = range / (duration / 16);
  let current = start;
  
  const timer = setInterval(() => {
    current += increment;
    if ((increment > 0 && current >= end) || (increment < 0 && current <= end)) {
      current = end;
      clearInterval(timer);
    }
    element.textContent = Math.round(current);
  }, 16);
}

// Initialize stat card animations on load
document.addEventListener('DOMContentLoaded', () => {
  const statValues = document.querySelectorAll('.stat-value');
  statValues.forEach(el => {
    const value = parseFloat(el.textContent) || 0;
    if (value > 0) {
      el.textContent = '0';
      setTimeout(() => animateValue(el, 0, value, 1000), 300);
    }
  });
});

// Floating Action Button (FAB) functionality
const fab = document.getElementById('fabRefresh');
if (fab) {
  fab.addEventListener('click', () => {
    showToast('🔄 Refreshing dashboard data...');
    
    // Add spinning animation
    fab.style.animation = 'spin 1s linear';
    
    // Reload the page after a brief delay
    setTimeout(() => {
      window.location.reload();
    }, 500);
  });
}

// Add spinning animation keyframe dynamically
const style = document.createElement('style');
style.textContent = `
  @keyframes spin {
    from { transform: rotate(0deg); }
    to { transform: rotate(360deg); }
  }
`;
document.head.appendChild(style);

// Note: The "Show more" button is handled above via renderTables()/applyRowLimit.
// We intentionally do NOT hide the entire insights wrapper here, so the preview
// remains visible on load and the button only expands/collapses rows.

// Smooth scroll to sections
function smoothScrollTo(target) {
  const element = document.querySelector(target);
  if (element) {
    element.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }
}

// Add table sorting functionality
document.querySelectorAll('th.sortable').forEach(th => {
  th.addEventListener('click', function() {
    const table = this.closest('table');
    const tbody = table.querySelector('tbody');
    const rows = Array.from(tbody.querySelectorAll('tr'));
    const index = Array.from(this.parentNode.children).indexOf(this);
    const isAsc = this.classList.contains('asc');
    
    // Remove all sorting classes
    table.querySelectorAll('th.sortable').forEach(h => {
      h.classList.remove('asc', 'desc');
    });
    
    // Add appropriate class
    this.classList.add(isAsc ? 'desc' : 'asc');
    
    // Sort rows
    rows.sort((a, b) => {
      const aValue = a.children[index].textContent.trim();
      const bValue = b.children[index].textContent.trim();
      
      const aNum = parseFloat(aValue);
      const bNum = parseFloat(bValue);
      
      if (!isNaN(aNum) && !isNaN(bNum)) {
        return isAsc ? bNum - aNum : aNum - bNum;
      }
      
      return isAsc ? 
        bValue.localeCompare(aValue) : 
        aValue.localeCompare(bValue);
    });
    
    // Reorder table
    rows.forEach(row => tbody.appendChild(row));
  });
});

// Enhanced tooltip positioning
document.querySelectorAll('.hint').forEach(hint => {
  hint.addEventListener('mouseenter', function(e) {
    const tooltip = window.getComputedStyle(this, '::after');
    const rect = this.getBoundingClientRect();
    
    // Check if tooltip would go off-screen
    if (rect.right + 340 > window.innerWidth) {
      this.style.setProperty('--tooltip-dir', 'left');
    }
  });
});

// Loading state management
function showLoadingState(container) {
  if (!container) return;
  container.innerHTML = `
    <div class="loading-skeleton">
      <div class="skeleton-row"></div>
      <div class="skeleton-row"></div>
      <div class="skeleton-row"></div>
    </div>
  `;
}

function hideLoadingState(container) {
  if (!container) return;
  const skeleton = container.querySelector('.loading-skeleton');
  if (skeleton) {
    skeleton.remove();
  }
}

// Exit button handled above via setMapOnly(false) on #exitMapOnlyBtn

// Responsive table handling
function makeTablesResponsive() {
  document.querySelectorAll('.minitable').forEach(table => {
    if (table.offsetWidth > table.parentElement.offsetWidth) {
      table.parentElement.style.overflowX = 'auto';
    }
  });
}

// Call on load and resize
window.addEventListener('load', makeTablesResponsive);
window.addEventListener('resize', makeTablesResponsive);

// Add keyboard shortcuts for refresh only (F key is handled above)
document.addEventListener('keydown', (e) => {
  // R key - refresh
  if (e.key === 'r' || e.key === 'R') {
    if (!e.target.matches('input, textarea') && e.ctrlKey) {
      e.preventDefault();
      const fab = document.getElementById('refreshDashboard');
      fab?.click();
    }
  }

  // Note: Escape to exit map-only is handled earlier via setMapOnly(false)
});

// Console welcome message
console.log("%c🌾 CropXcel Dashboard Enhanced", 
  "font-size: 16px; font-weight: bold; color: #0ea5e9; background: #f0f9ff; padding: 10px; border-radius: 8px;");
console.log("%cKeyboard Shortcuts:", "font-weight: bold; color: #64748b;");
console.log("%c  F - Toggle map fullscreen", "color: #94a3b8;");
console.log("%c  Ctrl+R - Refresh dashboard", "color: #94a3b8;");
console.log("%c  Escape - Exit fullscreen", "color: #94a3b8;");

console.log("✅ Dashboard enhancements loaded successfully");

// === Enhanced Dashboard Features ===

// FAB functionality
document.addEventListener('DOMContentLoaded', function() {
  const fab = document.getElementById('refreshDashboard');
  const fullscreenBtn = document.getElementById('fullscreenBtn');
  const refreshMapBtn = document.getElementById('refreshMapBtn');
  
  // Map fullscreen toggle
  if (fullscreenBtn) {
    fullscreenBtn.addEventListener('click', () => {
      try {
        const isCurrentlyMapOnly = document.body.classList.contains('map-only');
        const icon = fullscreenBtn.querySelector('i');
        
        // Add loading state
        fullscreenBtn.classList.add('loading');
        fullscreenBtn.disabled = true;
        
        // Add animation class based on current state
        if (isCurrentlyMapOnly) {
          fullscreenBtn.classList.add('compressing');
        } else {
          fullscreenBtn.classList.add('expanding');
        }
        
        // Trigger the mode change after a short delay for visual feedback
        setTimeout(() => {
          setMapOnly(!isCurrentlyMapOnly);
          
          // Update button icon and tooltip to indicate current state
          if (icon) {
            if (isCurrentlyMapOnly) {
              icon.className = 'fas fa-expand';
              fullscreenBtn.title = 'Enter Fullscreen Mode (F)';
              fullscreenBtn.setAttribute('aria-label', 'Enter fullscreen mode');
            } else {
              icon.className = 'fas fa-compress';
              fullscreenBtn.title = 'Exit Fullscreen Mode (Esc or F)';
              fullscreenBtn.setAttribute('aria-label', 'Exit fullscreen mode');
            }
          }
          
          // Remove loading and animation states
          setTimeout(() => {
            fullscreenBtn.classList.remove('loading', 'expanding', 'compressing');
            fullscreenBtn.disabled = false;
            
            // Brief success state
            fullscreenBtn.classList.add('success');
            setTimeout(() => {
              fullscreenBtn.classList.remove('success');
            }, 600);
            
            // Show success feedback
            showToast(isCurrentlyMapOnly ? 
              '📍 Normal view restored' : 
              '🗺️ Fullscreen map activated'
            );
          }, 100);
          
        }, 200);
        
      } catch (error) {
        console.error('Error toggling fullscreen:', error);
        fullscreenBtn.classList.remove('loading', 'expanding', 'compressing');
        fullscreenBtn.disabled = false;
        showToast('❌ Error toggling fullscreen mode');
      }
    });
    
    // Add keyboard navigation support
    fullscreenBtn.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' || e.key === ' ') {
        e.preventDefault();
        fullscreenBtn.click();
      }
    });
  }
  
  // FAB refresh with animation
  if (fab) {
    fab.addEventListener('click', function() {
      this.classList.add('spinning');
      showToast('Refreshing dashboard...');
      
      // Remove spinning class after animation
      setTimeout(() => {
        this.classList.remove('spinning');
        showToast('Dashboard refreshed');
      }, 1000);
      
      // Trigger page refresh or data reload
      setTimeout(() => {
        window.location.reload();
      }, 1200);
    });
  }
  
  // Map controls (fullscreen button handler is defined earlier in the file)
  
  if (refreshMapBtn) {
    refreshMapBtn.addEventListener('click', function() {
      const icon = this.querySelector('i');
      icon.style.transform = 'rotate(360deg)';
      
      // Reset rotation after animation
      setTimeout(() => {
        icon.style.transform = '';
      }, 500);
      
      // Refresh map overlay
      if (overlayLayer) {
        overlayLayer.redraw();
      }
      showToast('Map refreshed');
    });
  }
  
  // Animate stats on load
  animateStats();
});

// Animate stats values
function animateStats() {
  const statValues = document.querySelectorAll('.stat-value');
  
  statValues.forEach((stat, index) => {
    // Add entrance animation delay
    const card = stat.closest('.stat-card');
    if (card) {
      card.style.animationDelay = `${index * 0.1}s`;
    }
    
    // Animate numeric values only
    const text = stat.textContent.trim();
    const number = parseFloat(text);
    
    if (!isNaN(number) && number > 0) {
      stat.textContent = '0';
      animateValue(stat, 0, number, 800 + (index * 150));
    }
  });
}

// Animate numeric values with easing
function animateValue(element, start, end, duration) {
  const startTime = performance.now();
  const isDecimal = (end % 1 !== 0);
  
  function update(currentTime) {
    const elapsed = currentTime - startTime;
    const progress = Math.min(elapsed / duration, 1);
    
    // Easing function (ease-out)
    const eased = 1 - Math.pow(1 - progress, 3);
    const current = start + (end - start) * eased;
    
    if (isDecimal) {
      element.textContent = current.toFixed(1);
    } else {
      element.textContent = Math.floor(current);
    }
    
    if (progress < 1) {
      requestAnimationFrame(update);
    } else {
      element.textContent = isDecimal ? end.toFixed(1) : end;
    }
  }
  
  requestAnimationFrame(update);
}
