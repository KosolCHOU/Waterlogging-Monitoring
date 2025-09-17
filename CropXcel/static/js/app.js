// PLACE INTO: static/js/cropxcel.js

(function(){
  const classes = [
    { k:0, pct:52.8, ha:24.68, color:"#2ecc71" }, // Healthy
    { k:1, pct:29.8, ha:13.94, color:"#f1c40f" }, // Watch
    { k:2, pct:12.0, ha:5.60,  color:"#e67e22" }, // Concern
    { k:3, pct:5.5,  ha:2.56,  color:"#e74c3c" }  // Alert
  ];

  const segGroups = document.querySelectorAll('.scale-donut .segments .seg-ring');
  const sw = +document.querySelector('.scale-donut .segments').dataset.sw || 10;

  let totalHa = 0;
  classes.forEach(d => totalHa += d.ha);

  classes.forEach(d => {
    const el = document.querySelector(`.seg-ring[data-k="${d.k}"]`);
    if (!el) return;

    const r = +el.dataset.r;
    const C = 2 * Math.PI * r;
    const arc = (d.pct/100) * C;

    const circle = document.createElementNS('http://www.w3.org/2000/svg','circle');
    circle.setAttribute('cx',70);
    circle.setAttribute('cy',70);
    circle.setAttribute('r',r);
    circle.setAttribute('stroke',d.color);
    circle.setAttribute('stroke-width',sw);
    circle.setAttribute('fill','none');
    circle.setAttribute('class','seg');

    circle.setAttribute('stroke-dasharray',`${arc} ${C-arc}`);
    circle.setAttribute('stroke-dashoffset',C*0.25); // start at 12 o’clock

    el.appendChild(circle);
  });

  // update center label
  const label = document.querySelector('#donutCenter');
  if (label) label.textContent = `${totalHa.toFixed(2)} Ha`;
})();

/* ================= UI (tables, buttons, map-only, probe) ================= */
(function(){
  function initUI(){
    try {
      const farmerHTML = `$FARMER`;
      const techHTML   = `$TECH`;

      const tableWrap = document.getElementById('insightsTable');
      const title   = document.getElementById('insTitle');
      const moreBtn = document.getElementById('toggleInsights');
      const techBtn = document.getElementById('techBtn');
      const mapBtn  = document.getElementById('mapOnlyBtn');
      const exitBtn = document.getElementById('exitMapOnlyBtn');

      const FARMER_MAX = Number(tableWrap?.dataset?.max ?? 5);
      const TECH_MAX   = Number(tableWrap?.dataset?.techMax ?? 10);

      let techMode = false;
      let expandedFarmer = false;
      let expandedTech   = false;

      function applyRowLimit(t, max){
        if (!t || !t.tBodies[0]) return;
        const rows = Array.from(t.tBodies[0].rows);
        rows.forEach((r,i)=>{ r.style.display = (i < max) ? '' : 'none'; });
      }

      function fixFarmerTable(){
        if (techMode) return;
        const t = tableWrap.querySelector('table'); if (!t) return;
        t.classList.add('minitable','farmer');
        if (!t.querySelector('colgroup')) {
          const cg = document.createElement('colgroup');
          ['90px','90px','60px','70px','auto'].forEach(w=>{
            const c = document.createElement('col'); c.style.width = w; cg.appendChild(c);
          });
          t.insertBefore(cg, t.firstChild);
        }
      }
      function fixTechTable(){
        if (!techMode) return;
        const t = tableWrap.querySelector('table'); if (!t) return;
        t.classList.add('minitable','tech');
        if (!t.querySelector('colgroup')) {
          const cg = document.createElement('colgroup');
          ['100px','80px','80px','80px','80px','80px','80px','80px'].forEach(w=>{
            const c = document.createElement('col'); c.style.width = w; cg.appendChild(c);
          });
          t.insertBefore(cg, t.firstChild);
        }
      }

      function makeSortableAndSparky(t){
        if (!t) return;
        const ths = t.querySelectorAll('thead th');
        ths.forEach((th, idx)=>{
          th.classList.add('sortable');
          th.onclick = () => sortByColumn(t, idx, th);
        });

        if (techMode && t.tBodies[0] && ths.length){
          const rows = Array.from(t.tBodies[0].rows);
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
              let pct = 0;
              if (Number.isFinite(v) && max[c] > min[c]) pct = (v-min[c])/(max[c]-min[c])*100;
              cell.classList.add('spark');
              cell.style.setProperty('--pct', pct.toFixed(2));
            }
          });
        }
      }

      function sortByColumn(table, colIdx, thEl, dir){
        const tbody = table.tBodies[0];
        const rows  = Array.from(tbody.rows);
        const asc = (dir === 'asc') ? true : (dir === 'desc') ? false : !thEl.classList.contains('asc');

        const parseVal = (s)=>{
          const txt = (s||'').trim();
          if (/^\d{4}-\d{2}-\d{2}$/.test(txt)) return new Date(txt).getTime();
          const n = parseFloat(txt.replace(/[, ]/g,''));
          return Number.isFinite(n) ? n : txt;
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

      function render(){
        if (!tableWrap) return;

        if (techMode) {
          title.textContent = 'Technical Details';
          tableWrap.innerHTML = techHTML;
          moreBtn.style.display = '';
          moreBtn.textContent = expandedTech ? 'Show less' : 'Show more';
          techBtn.textContent = 'Exit Technical Details';
          document.body.classList.add('tech-mode');
        } else {
          title.textContent = 'Per-pass Insights';
          tableWrap.innerHTML = farmerHTML;
          moreBtn.style.display = '';
          moreBtn.textContent = expandedFarmer ? 'Show less' : 'Show more';
          techBtn.textContent = 'Show Technical Details';
          document.body.classList.remove('tech-mode');
        }

        fixFarmerTable();
        fixTechTable();

        const t = tableWrap.querySelector('table');
        if (t){
          makeSortableAndSparky(t);
          const dateTH = t.querySelector('thead th:first-child');
          if (dateTH) sortByColumn(t, 0, dateTH, 'desc');

          if (!techMode && !expandedFarmer) applyRowLimit(t, FARMER_MAX);
          if ( techMode && !expandedTech  ) applyRowLimit(t, TECH_MAX);
        }

        requestAnimationFrame(()=> window.tagDonutRings && window.tagDonutRings());
        window.__refreshMap && window.__refreshMap();
        window.rebuildDonut && window.rebuildDonut();
        window.tagDonutRings && window.tagDonutRings();
      }

      render();

      moreBtn.onclick = ()=>{
        if (techMode) expandedTech = !expandedTech;
        else          expandedFarmer = !expandedFarmer;
        render();
      };
      techBtn.onclick = ()=>{ techMode = !techMode; render(); };

      function setMapOnly(on){
        document.body.classList.toggle('map-only', !!on);
        const isOn = document.body.classList.contains('map-only');
        mapBtn.textContent  = isOn ? 'Exit Map Only' : 'Map Only';
        exitBtn.style.display = isOn ? '' : 'none';
        window.__refreshMap && window.__refreshMap();
      }
      mapBtn.onclick  = ()=> setMapOnly(!document.body.classList.contains('map-only'));
      exitBtn.onclick = ()=> setMapOnly(false);
      document.addEventListener('keydown', (e)=>{ if (e.key === 'Escape') setMapOnly(false); });

    } catch (err) {
      console.error('UI init failed:', err);
      window.__refreshMap && window.__refreshMap();
    }
  }

  if (document.readyState === 'loading'){
    document.addEventListener('DOMContentLoaded', initUI);
  } else {
    initUI();
  }
})();

(function(){
  const card = document.getElementById('adviceCard');
  if (!card) return;

  const btn   = document.getElementById('adviceToggle');
  const text  = document.getElementById('adviceText');
  const sub   = document.getElementById('adviceSub');
  const pill  = document.getElementById('adviceLevel');

  // Read metrics
  const s1   = parseFloat(card.dataset.s1 || '0') || 0;
  const im24 = parseFloat(card.dataset.imerg24 || '0') || 0;
  const fc48 = parseFloat(card.dataset.fc48 || '0') || 0;
  const lvl  = String(card.dataset.level || 'GREEN').toUpperCase();

  // Read messages (JSON strings)
  const FARMER = card.dataset.farmer ? JSON.parse(card.dataset.farmer) : "";
  const TECH   = card.dataset.tech   ? JSON.parse(card.dataset.tech)   : "";

  // Level styling
  card.classList.remove('green','yellow','red');
  if (lvl === 'RED')    card.classList.add('red');
  if (lvl === 'YELLOW') card.classList.add('yellow');
  if (lvl === 'GREEN')  card.classList.add('green');
  if (pill) pill.textContent = lvl;

  // Toggle
  let showTech = false;
  function render(){
    text.textContent = showTech ? TECH : FARMER;
    sub.textContent  = `(${s1.toFixed(2)} ha water | 24h: ${im24.toFixed(1)} mm | 48h: ${fc48.toFixed(1)} mm)`;
    btn.textContent  = showTech ? "Show Farmer" : "Show Technical";
  }
  btn?.addEventListener('click', ()=>{ showTech = !showTech; render(); });
  render();
})();

(function(){
  const btn  = document.getElementById('adviceToggle');
  const text = document.getElementById('adviceText');

  let showTech = false;
  function render(){
    text.textContent = showTech ? TECH : FARMER;
    btn.textContent  = showTech ? "Show Farmer" : "Show Technical";
  }

  btn && btn.addEventListener('click', ()=>{ showTech = !showTech; render(); });
  render();

  // Optional: also reflect level with classes for styling
  const root = document.getElementById('adviceCard');
  const lvl  = root ? root.getAttribute('data-level') : null;
  if (root && lvl){
    root.classList.remove('is-red','is-yellow','is-green');
    if (lvl==="RED") root.classList.add('is-red');
    if (lvl==="YELLOW") root.classList.add('is-yellow');
    if (lvl==="GREEN") root.classList.add('is-green');
  }
})();

(function(){
  function getMap(){
    const el=document.querySelector('div[id^="map_"]');
    return el ? (window[el.id.replace(/-/g,'_')] || null) : null;
  }
  function refresh(){ const m=getMap(); if (m && m.invalidateSize) setTimeout(()=>m.invalidateSize(true), 250); }
  function start(){ refresh(); window.tagDonutRings && window.tagDonutRings(); }

  if (document.readyState === 'loading'){
    document.addEventListener('DOMContentLoaded', start);
  } else {
    start();
  }
})();

var map = L.map('map').setView([11.45, 105.42], 15);
L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    maxZoom: 22
}).addTo(map);

var drawnItems = new L.FeatureGroup();
map.addLayer(drawnItems);

var drawControl = new L.Control.Draw({
    draw: {
        polygon: { shapeOptions: { color: "#ff7800" } },
        rectangle: { shapeOptions: { color: "#22c55e" } },
        circle: false,
        marker: false,
        polyline: false
    },
    edit: {
        featureGroup: drawnItems
    }
});
map.addControl(drawControl);

map.on(L.Draw.Event.CREATED, function (e) {
    var layer = e.layer;
    drawnItems.addLayer(layer);

    // auto-fill hidden input with GeoJSON
    var geojson = layer.toGeoJSON();
    document.getElementById("geojsonInput").value = JSON.stringify(geojson);
});
