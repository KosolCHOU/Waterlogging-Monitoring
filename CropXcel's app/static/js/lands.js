// CSRF
function getCookie(name){
  const v = `; ${document.cookie}`.split(`; ${name}=`); 
  if (v.length === 2) return v.pop().split(';').shift();
}
const csrftoken = getCookie('csrftoken');

async function getJSON(url){
  const r = await fetch(url, {cache:'no-store'});
  if(!r.ok) throw new Error(await r.text());
  return r.json();
}
// Read Django URL from the template (with a safe fallback)
const SAVE_URL = document.getElementById('drawer')?.dataset.saveUrl || "/aoi/upload/";

// Safer error handling (don’t dump whole HTML into the UI)
async function postJSON(url, body){
  const res  = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json", "X-CSRFToken": csrftoken },
    body: JSON.stringify(body)
  });
  const text = await res.text();
  if (!res.ok) {
    // Trim long HTML responses and surface status
    const brief = text.replace(/<[^>]+>/g,"").slice(0, 200).trim();
    throw new Error(`${res.status} ${res.statusText}${brief ? ` — ${brief}` : ""}`);
  }
  try { return JSON.parse(text); } catch { throw new Error("Invalid JSON from server."); }
}

// Field cards
const fieldsList = document.getElementById('fieldsList');
const emptyMsg   = document.getElementById('emptyMsg');

function fieldCardHtml(f){
  const id = f.id ?? f.field_id;
  const name = f.name || `Field ${id}`;
  const created = (f.created_at || f.created || "").toString().slice(0,10);
  const reportHref = `/dashboard/${id}/`;
  const recHref    = `/fields/${id}/recommend/`;
  const areaTxt = (f.area_ha != null) ? `${Number(f.area_ha).toFixed(2)} ha` : "—";
  return `
    <div class="card" data-id="${id}">
      <h4>${name}</h4>
      <div class="row">
        <span class="pill"><span id="area-${id}">${areaTxt}</span></span>
        <span class="muted">${created || ""}</span>
      </div>
      <div style="display:flex; gap:8px; margin-top:8px;">
        <button class="go" onclick="setLastField(${id});location.href='${reportHref}'">See Field Report</button>
      </div>
    </div>`;
}

function setLastField(id){
  localStorage.setItem("lastFieldId", id);
  document.cookie = "last_field=" + id + "; path=/; max-age=" + (60*60*24*30);
}

async function loadFields(){
  try{
    const j = await getJSON("/api/fields/?ordering=-id");
    const items = Array.isArray(j) ? j : (j.results || []);
    window._fieldsCache = items;
    fieldsList.innerHTML = items.map(fieldCardHtml).join("");
    emptyMsg.style.display = items.length ? "none" : "block";
    fillAreasIfMissing(items);
  }catch(e){
    fieldsList.innerHTML = `<div class="muted">Failed to load fields: ${e.message}</div>`;
  }
}

async function fillAreasIfMissing(items){
  for(const f of items){
    const id = f.id ?? f.field_id;
    const areaSpan = document.getElementById(`area-${id}`);
    if(!areaSpan) continue;
    const already = areaSpan.textContent && areaSpan.textContent !== "—";
    if (already) continue;
    try{
      const detail = await getJSON(`/api/fields/${id}/`);
      const geom = detail.geom || detail.geometry || (detail.field?.geom);
      if(geom){
        const m2 = turf.area({ type:"Feature", geometry: geom, properties:{} });
        const ha = m2 / 10000.0;
        areaSpan.textContent = `${ha.toFixed(2)} ha`;
      }
    }catch(_){}
  }
}

// Drawer
const drawer = document.getElementById('drawer');
const mask   = document.getElementById('drawerMask');
const openBtn= document.getElementById('addFieldBtn');
const closeBtn=document.getElementById('closeDrawer');
function openDrawer(){ drawer.classList.add('open'); mask.classList.add('open'); initMapIfNeeded(); }
function closeDrawer(){ drawer.classList.remove('open'); mask.classList.remove('open'); }
openBtn.addEventListener('click', openDrawer);
closeBtn.addEventListener('click', closeDrawer);
mask.addEventListener('click', closeDrawer);

// Map + draw
let map = null, drawnItems = null;
let LAST = null;
const areaVal   = () => document.getElementById('areaVal');
const areaInput = () => document.getElementById('areaInput');
const saveBtn   = () => document.getElementById('saveBtn');
const saveMsg   = () => document.getElementById('saveMsg');

function hectaresFromGeoJSON(gj){
  if(!gj) return 0;
  try{ if(['Polygon','MultiPolygon'].includes(gj.geometry?.type)) return turf.area(gj)/10000.0; }catch(_){}
  return 0;
}
function updateAreaDisplay(gj){
  const ha = hectaresFromGeoJSON(gj);
  const txt = `${ha.toFixed(2)} hectares`;
  areaVal().textContent = txt; areaInput().value = txt;
}

function initMapIfNeeded(){
  if (map) return;
  map = L.map('map').setView([11.45, 105.42], 15);
  const esri = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', { attribution: '© Esri' }).addTo(map);
  const osm  = L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', { attribution: '© OSM' });
  drawnItems = new L.FeatureGroup().addTo(map);
  map.addControl(new L.Control.Draw({
    draw: {
      polygon:{ showArea:true, allowIntersection:false, shapeOptions:{ color:'#10b981', weight:2 } },
      rectangle:{ shapeOptions:{ color:'#22c55e', weight:2 } },
      polyline:false, circle:false, circlemarker:false, marker:false
    },
    edit:{ featureGroup: drawnItems, remove:true }
  }));

  let currentBase = "satellite";  // default
  document.getElementById('layersBtn').onclick = () => {
    if (currentBase === "satellite") {
      map.removeLayer(esri);
      osm.addTo(map);
      currentBase = "street";
      document.getElementById('layersBtn').textContent = "🌍 Satellite View";
    } else {
      map.removeLayer(osm);
      esri.addTo(map);
      currentBase = "satellite";
      document.getElementById('layersBtn').textContent = "🗺️ Street Map";
    }
  };

  document.getElementById('locBtn').onclick = ()=>{
    if(!navigator.geolocation) return;
    navigator.geolocation.getCurrentPosition(pos=>{
      map.setView([pos.coords.latitude, pos.coords.longitude], 16);
    });
  };

  map.on(L.Draw.Event.CREATED, (e)=>{ drawnItems.clearLayers(); drawnItems.addLayer(e.layer); LAST = e.layer.toGeoJSON(); updateAreaDisplay(LAST); });
  map.on(L.Draw.Event.EDITED, (e)=>{ const layers=e.layers.getLayers(); if(layers.length){ LAST=layers[0].toGeoJSON(); updateAreaDisplay(LAST);} });
  map.on(L.Draw.Event.DELETED, ()=>{ LAST=null; updateAreaDisplay(null); saveMsg().textContent=''; });
}

async function saveField(){
  if(!LAST){ saveMsg().textContent = "⚠️ Draw a polygon or rectangle first."; return; }
  saveBtn().disabled = true; saveMsg().textContent = "Saving…";
  try{
    const nameVal = document.getElementById('nameInput').value.trim();

    // 🔽 use the Django URL from the template
    const aux = await postJSON(SAVE_URL, { feature: LAST, name: nameVal || null });

    const fieldId = aux.field_id;
    if(!fieldId) throw new Error("Server did not return field_id.");
    const sHa = (aux.area_ha ?? hectaresFromGeoJSON(LAST)).toFixed(2);
    saveMsg().textContent = `✅ Saved. Server area: ${sHa} ha — analysis started.`;
    await loadFields();
    setTimeout(closeDrawer, 500);
  }catch(e){
    saveMsg().textContent = "❌ Save failed: " + e.message;
  }finally{
    saveBtn().disabled = false;
  }
}
document.getElementById('saveBtn').addEventListener('click', saveField);

// boot
loadFields();