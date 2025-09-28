(async function(){
  // read field id from the container's data attribute (no Django templating in JS)
  const root = document.querySelector('.wrap');
  const fieldId = root ? Number(root.dataset.fieldId) : null;

  try { document.cookie = `last_field=${fieldId}; Max-Age=2592000; Path=/; SameSite=Lax`; } catch(e) {}

  if (!fieldId) {
    console.warn("No field id found on .wrap[data-field-id]");
    return;
  }

  const res = await fetch(`/fields/${fieldId}/forecast.json`, {cache:'no-store'});
  const data = await res.json();
  if(!data.ok) return;

  // when/where
  const now = new Date();
  document.getElementById('when').textContent =
    now.toLocaleString('en-GB',{weekday:'long',day:'numeric',month:'short',hour:'2-digit',minute:'2-digit'});

  // today
  const t = data.today || {};
  const todayRow = data.daily?.[0] || {};
  const hour = now.getHours();
  let desc;
  if (t.rain_mm > 5) desc = "Rainy";
  else if (t.prob_max > 50) desc = (hour>=18||hour<6) ? "Cloudy Night" : "Mostly Cloudy";
  else desc = (hour>=18||hour<6) ? "Clear Night" : "Sunny";
  const icon = desc.includes("Rain") ? "🌧️" :
               desc.includes("Cloudy") ? (hour>=18||hour<6 ? "🌙☁️" : "☁️") :
               (hour>=18||hour<6 ? "🌙" : "☀️");
  document.getElementById('temp').textContent = `${Math.round(todayRow.tmax || 0)}°`;
  document.getElementById('desc').textContent = desc;
  document.getElementById('icon').textContent = icon;
  document.getElementById('iconBig').textContent = icon;
  document.getElementById('rightCloud').textContent = desc.includes("Cloudy") ? desc : (desc.includes("Rain") ? "Rainy" : desc);
  document.getElementById('rightRain').textContent = `Rain – ${(t.prob_max ?? 0)}%`;
  document.getElementById('chipProb').textContent = `${t.prob_max ?? "—"}%`;
  document.getElementById('chipRain').textContent = `${(t.rain_mm ?? 0).toFixed(1)} mm`;
  document.getElementById('chipWind').textContent = `${Math.round(t.wind_max ?? 0)} km/h`;

  // 7-day table data array
  const tbody = document.querySelector('#dailyTbl tbody');
  const rows = (data.daily||[]).map(d=>({date:d.date,rain_mm:d.rain_mm,prob_max:d.prob_max,tmin:d.tmin,tmax:d.tmax,wind_max:d.wind_max}));

  // risk classifier (rain-first)
  function classifyRisk(rain, prob){
    if (rain < 3) return {label:'Low', cls:'low'};
    const p = Math.max(0, Math.min(100, prob||0))/100;
    const score = rain * Math.pow(p, 0.9);
    if ( (rain >= 25 && prob >= 60) || score >= 25 ) return {label:'High', cls:'high'};
    if ( (rain >= 8  && prob >= 40) || score >= 10 ) return {label:'Moderate', cls:'mod'};
    return {label:'Low', cls:'low'};
  }

  function paint(){
    tbody.innerHTML = rows.map(d=>{
      const rain = +(d.rain_mm ?? 0);
      const prob = Math.round(d.prob_max ?? 0);
      const wind = Math.round(d.wind_max ?? 0);
      const tmin = Math.round(d.tmin ?? 0), tmax = Math.round(d.tmax ?? 0);

      const dt = new Date(d.date);
      const day = dt.toLocaleDateString('en-GB',{weekday:'short', day:'numeric', month:'short'});
      let iconDay = "☀️";
      if (rain >= 30 && prob >= 60)      iconDay = "⛈️";
      else if (rain >= 10)               iconDay = "🌧️";
      else if (wind >= 25 && rain < 3)   iconDay = "🌬️";
      else if (tmax >= 35)               iconDay = "🔥";
      else if (tmax <= 25)               iconDay = "❄️";
      else if (prob >= 50)               iconDay = "☁️";

      let rainCls = 'rain-low', rainIcon='🌱';
      if (rain >= 30){ rainCls='rain-high'; rainIcon='🌧️'; }
      else if (rain >= 10){ rainCls='rain-mod';  rainIcon='☔'; }

      let probBg='prob-low-bg', probIcon='📉';
      if (prob >= 70){ probBg='prob-high-bg'; probIcon='⚡'; }
      else if (prob >= 40){ probBg='prob-med-bg'; probIcon='☁️'; }

      let tempBg='temp-warm-bg', tempIcon='🌡️';
      if (tmax >= 35){ tempBg='temp-hot-bg';  tempIcon='🔥'; }
      else if (tmax <= 25){ tempBg='temp-cool-bg'; tempIcon='❄️'; }

      let windBg='', windIcon='🍃';
      if (wind >= 35){ windBg='wind-high-bg'; windIcon='💨'; }
      else if (wind >= 20){ windBg='wind-med-bg'; windIcon='🌬️'; }

      const risk = classifyRisk(rain, prob);

      return `
        <tr title="Rain: ${rain.toFixed(1)} mm | Prob.max: ${prob}% | Wind: ${wind} km/h | Temp: ${tmin}–${tmax}°C">
          <td class="date">${iconDay} ${day}</td>
          <td class="${rainCls}">${rainIcon} ${rain.toFixed(1)}</td>
          <td class="${probBg}">${probIcon} ${prob}%</td>
          <td class="${tempBg}">${tempIcon} ${tmin}–${tmax}°C</td>
          <td class="${windBg}">${windIcon} ${wind} km/h</td>
          <td><span class="pill ${risk.cls}">${risk.label}</span></td>
        </tr>`;
    }).join('');
  }

  paint();

  function niceTicks(maxVal, count=5){
    const nice = (x)=>{ const exp=Math.pow(10, Math.floor(Math.log10(x||1)));
                        const f=x/exp; const nf=(f<1.5?1:(f<3?2:(f<7?5:10))); return nf*exp; };
    const top = nice(maxVal);
    const step = nice(top/(count-1));
    const ticks = [];
    for(let v=0; v<=top+1e-9; v+=step){ ticks.push(Math.round(v*10)/10); }
    if (ticks[ticks.length-1] !== Math.round(top*10)/10) ticks.push(Math.round(top*10)/10);
    return ticks;
  }

  let showProb = true, showRain = true;
  function render72(series){
    const svg  = document.getElementById('svg72');
    const wrap = document.getElementById('plot72');
    const tip  = document.getElementById('tip72');

    if (!tip.classList.contains('plot-tip')) tip.className = 'plot-tip';
    if (!Array.isArray(series) || !series.length) {
      svg.innerHTML = '';
      tip.style.display = 'none';
      return;
    }

    svg.innerHTML = '';
    const m = { top: 10, right: 44, bottom: 20, left: 44 };
    const W = Math.max(200, wrap.clientWidth  || 600);
    const H = Math.max(140, wrap.clientHeight || 180);
    svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
    const iw = Math.max(10, W - m.left - m.right);
    const ih = Math.max(10, H - m.top  - m.bottom);
    const n = series.length;
    const step = iw / Math.max(1, n);
    const bwP  = step * 0.35;
    const bwR  = step * 0.65;
    const x = i => m.left + i * step + step/2;

    function niceTicks(maxVal, count=5){
      const nice = (x)=>{ const exp=Math.pow(10, Math.floor(Math.log10(x||1)));
                          const f=x/exp; const nf=(f<1.5?1:(f<3?2:(f<7?5:10))); return nf*exp; };
      const top = nice(maxVal);
      const step = nice(top/(count-1));
      const ticks = [];
      for(let v=0; v<=top+1e-9; v+=step){ ticks.push(Math.round(v*10)/10); }
      if (ticks[ticks.length-1] !== Math.round(top*10)/10) ticks.push(Math.round(top*10)/10);
      return ticks;
    }

    const maxRainData = Math.max(1, ...series.map(d => +d.rain || 0));
    const rainTicks = niceTicks(maxRainData, 5);
    const maxRain   = rainTicks[rainTicks.length - 1];

    const yProb = p => m.top + (1 - (Math.max(0, Math.min(100, +p || 0)) / 100)) * ih;
    const yRain = r => m.top + (1 - ((+r || 0) / maxRain)) * ih;

    const defs = document.createElementNS('http://www.w3.org/2000/svg','defs');
    defs.innerHTML = `
      <linearGradient id="rainGrad72" x1="0" y1="1" x2="0" y2="0">
        <stop offset="0%"  stop-color="#6ee7b7"/>
        <stop offset="100%" stop-color="#34d399"/>
      </linearGradient>`;
    svg.appendChild(defs);

    const gGrid = document.createElementNS('http://www.w3.org/2000/svg','g');
    gGrid.setAttribute('class','grid');
    [0,25,50,75,100].forEach(v=>{
      const y = yProb(v);
      const ln = document.createElementNS('http://www.w3.org/2000/svg','line');
      ln.setAttribute('x1', m.left); ln.setAttribute('x2', W - m.right);
      ln.setAttribute('y1', y);      ln.setAttribute('y2', y);
      gGrid.appendChild(ln);
    });
    svg.appendChild(gGrid);

    const gBars = document.createElementNS('http://www.w3.org/2000/svg','g');
    series.forEach((d,i)=>{
      const cx = x(i);
      const rVal = +d.rain || 0;
      const rY   = yRain(rVal);
      const rH   = Math.max(0, ih - (rY - m.top));
      const r    = document.createElementNS('http://www.w3.org/2000/svg','rect');
      r.setAttribute('class', 'barRain');
      r.style.opacity = showRain ? 1 : 0.12;
      r.setAttribute('x',      cx - bwR/2);
      r.setAttribute('y',      rY);
      r.setAttribute('width',  Math.max(1, bwR - 1));
      r.setAttribute('height', rH);
      gBars.appendChild(r);

      const pVal = +d.prob || 0;
      const pY   = yProb(pVal);
      const pH   = Math.max(0, ih - (pY - m.top));
      const p    = document.createElementNS('http://www.w3.org/2000/svg','rect');
      p.setAttribute('class', 'barProb');
      p.style.opacity = showProb ? .65 : 0.12;
      p.setAttribute('x',      cx - bwP/2);
      p.setAttribute('y',      pY);
      p.setAttribute('width',  Math.max(1, bwP));
      p.setAttribute('height', pH);
      gBars.appendChild(p);
    });
    svg.appendChild(gBars);

    const gAxis = document.createElementNS('http://www.w3.org/2000/svg','g');
    gAxis.setAttribute('class','axis');
    [0,25,50,75,100].forEach(v=>{
      const t = document.createElementNS('http://www.w3.org/2000/svg','text');
      t.setAttribute('x', m.left - 6);
      t.setAttribute('y', yProb(v) + 3);
      t.setAttribute('text-anchor','end');
      t.textContent = v + '%';
      gAxis.appendChild(t);
    });
    rainTicks.forEach(v=>{
      const t = document.createElementNS('http://www.w3.org/2000/svg','text');
      t.setAttribute('x', W - m.right + 6);
      t.setAttribute('y', yRain(v) + 3);
      t.setAttribute('text-anchor','start');
      t.textContent = v + 'mm';
      gAxis.appendChild(t);
    });
    series.forEach((d,i)=>{
      if (i % 6 !== 0) return;
      const tx = document.createElementNS('http://www.w3.org/2000/svg','text');
      tx.setAttribute('x', x(i));
      tx.setAttribute('y', H - 4);
      tx.setAttribute('text-anchor','middle');
      tx.textContent = d.h || '';
      gAxis.appendChild(tx);
    });
    svg.appendChild(gAxis);

    const gHover = document.createElementNS('http://www.w3.org/2000/svg','g');
    svg.appendChild(gHover);

    const band = document.createElementNS('http://www.w3.org/2000/svg','rect');
    band.setAttribute('class','hoverBand');
    band.setAttribute('y', m.top);
    band.setAttribute('height', ih);
    band.setAttribute('width', Math.max(6, step));
    band.style.display = 'none';
    band.style.pointerEvents = 'none';
    gHover.appendChild(band);

    const crossHalo = document.createElementNS('http://www.w3.org/2000/svg','line');
    crossHalo.setAttribute('y1', m.top);
    crossHalo.setAttribute('y2', H - m.bottom);
    crossHalo.setAttribute('stroke', '#ffffff');
    crossHalo.setAttribute('stroke-opacity', '0.9');
    crossHalo.setAttribute('stroke-width', '4');
    crossHalo.setAttribute('vector-effect', 'non-scaling-stroke');
    crossHalo.style.display = 'none';
    crossHalo.style.pointerEvents = 'none';
    gHover.appendChild(crossHalo);

    const cross = document.createElementNS('http://www.w3.org/2000/svg','line');
    cross.setAttribute('class','cross');
    cross.setAttribute('y1', m.top);
    cross.setAttribute('y2', H - m.bottom);
    cross.setAttribute('stroke', '#111827');
    cross.setAttribute('stroke-width', '2');
    cross.setAttribute('vector-effect', 'non-scaling-stroke');
    cross.style.display = 'none';
    cross.style.pointerEvents = 'none';
    gHover.appendChild(cross);

    const clamp = (v, a, b) => Math.max(a, Math.min(b, v));

    function showAtIndex(i){
      if (Number.isNaN(i)) return;
      i = clamp(i, 0, n - 1);
      const cx = x(i);

      band.setAttribute('x', cx - (Math.max(6, step)/2));
      band.style.display = 'block';

      crossHalo.setAttribute('x1', cx);
      crossHalo.setAttribute('x2', cx);
      crossHalo.style.display = 'block';

      const crossLine = cross;
      crossLine.setAttribute('x1', cx);
      crossLine.setAttribute('x2', cx);
      crossLine.style.display = 'block';

      const d = series[i];
      const tip = document.getElementById('tip72');
      tip.textContent = `${d.h || ''} • ${Math.round(+d.prob||0)}% • ${(+d.rain||0)} mm`;
      tip.style.display = 'block';

      tip.style.left = `${cx}px`;
      tip.style.top  = `${m.top + 6}px`;
      const tipW = tip.offsetWidth || 0;
      let left = cx, half = tipW/2;
      if (left - half < m.left)      left = m.left + half + 2;
      if (left + half > W - m.right) left = W - m.right - half - 2;
      tip.style.left = `${left}px`;
    }

    function onMove(evt){
      const rect = svg.getBoundingClientRect();
      const relX = evt.clientX - rect.left - m.left;
      const i    = Math.round(relX / step - 0.5);
      showAtIndex(i);
    }

    svg.onmousemove  = onMove;
    svg.onmouseleave = ()=>{
      const tip = document.getElementById('tip72');
      tip.style.display = 'none';
      cross.style.display = 'none';
      crossHalo.style.display = 'none';
      band.style.display = 'none';
    };

    if (window._ro72) window._ro72.disconnect();
    if (window.ResizeObserver){
      let rafId = null;
      window._ro72 = new ResizeObserver(()=>{
        if (rafId) cancelAnimationFrame(rafId);
        rafId = requestAnimationFrame(()=> render72(series));
      });
      window._ro72.observe(wrap);
    }
  }

  const series72 = Array.isArray(data.hourly72) ? data.hourly72 : [];
  render72(series72);

  document.getElementById('legend72').addEventListener('click', (e)=>{
    const chip = e.target.closest('.chip'); if(!chip) return;
    const key = chip.dataset.series;
    if (key === 'prob') showProb = !showProb;
    if (key === 'rain') showRain = !showRain;
    chip.classList.toggle('off'); chip.setAttribute('aria-pressed', chip.classList.contains('off') ? 'false' : 'true');
    render72(series72);
  });

  // Sorting
  document.querySelectorAll('#dailyTbl thead th').forEach(th=>{
    th.addEventListener('click',()=>{
      const k = th.dataset.k; if(!k) return;
      const asc = !th.classList.toggle('desc');
      document.querySelectorAll('#dailyTbl thead th').forEach(h=>h!==th&&h.classList.remove('desc'));
      rows.sort((a,b)=>{
        if(k==='t') return asc ? (a.tmax-b.tmax) : (b.tmax-a.tmax);
        if(k==='date') return asc ? a.date.localeCompare(b.date) : b.date.localeCompare(a.date);
        return asc ? ((a[k]??0)-(b[k]??0)) : ((b[k]??0)-(a[k]??0));
      });
      paint();
    });
  });

  // KPIs (rain-first)
  const mm = data.rain_mm_72h || 0;
  const peak = data.precip_prob_peak_72h || 0;
  const peakTime = data.precip_prob_peak_time || '--:--';
  const kpiRisk = classifyRisk(mm, peak).label;
  document.getElementById('kpiRain').textContent = `${mm} mm`;
  document.getElementById('kpiPeak').textContent = `${peak}%`;
  document.getElementById('kpiTime').textContent = peakTime;
  document.getElementById('kpiRisk').textContent = kpiRisk;

  // Recommendations
  const adviceWrap = document.getElementById("adviceWrap");
  const advs  = Array.isArray(data.advisories) ? data.advisories : [];
  const flags = Array.isArray(data.flags)       ? data.flags       : [];

  const iconMap = {
    heavy_rain_soon:"🌧️", wet_spell:"☔", flooding:"🌊",
    wind_risk:"💨", heat_wave:"🔥", cool_spell:"❄️",
    dry_spell:"🌵", default:"🌱"
  };

  function inferFlag(text){
    const t=(text||"").toLowerCase();
    if(/flood|waterlogging|inundat/.test(t)) return "flooding";
    if(/heavy rain|downpour|thunder/.test(t)) return "heavy_rain_soon";
    if(/wet spell|many days rain|prolonged rain/.test(t)) return "wet_spell";
    if(/wind|gust|gale/.test(t)) return "wind_risk";
    if(/heat|hot|heatwave/.test(t)) return "heat_wave";
    if(/cold|cool|chill/.test(t)) return "cool_spell";
    if(/dry|drought/.test(t)) return "dry_spell";
    return "default";
  }

  function beautify(text, flag, downgradedFrom=null){
    if (flag === "default" && downgradedFrom) {
      if (downgradedFrom === "heavy_rain_soon")
        return "<b>Action:</b> Continue normal farm work.<br><b>Why:</b> No heavy rain expected in the next 72h.";
      if (downgradedFrom === "wet_spell")
        return "<b>Action:</b> Manage irrigation as usual.<br><b>Why:</b> No prolonged wet spell in the next 7 days.";
      if (downgradedFrom === "flooding")
        return "<b>Action:</b> No flood-specific action required.<br><b>Why:</b> No day has flood-level rain forecast.";
      return "<b>Action:</b> No special action needed.<br><b>Why:</b> Risks remain low.";
    }
    switch(flag){
      case "flooding":
        return "<b>Action:</b> Clear field drains, check bunds.<br><b>Why:</b> High rainfall (≥30 mm) may cause localized flooding.";
      case "heavy_rain_soon":
        return "<b>Action:</b> Delay fertilizer and field work.<br><b>Why:</b> Heavy rain likely within 72h, risk of wash-off.";
      case "wet_spell":
        return "<b>Action:</b> Avoid fertilizer application; reduce irrigation.<br><b>Why:</b> Continuous rain ≥3 days increases waterlogging risk.";
      case "wind_risk":
        return "<b>Action:</b> Secure seedlings and structures.<br><b>Why:</b> Strong winds expected may damage young crops.";
      case "heat_wave":
        return "<b>Action:</b> Irrigate in mornings/evenings; apply mulch.<br><b>Why:</b> Very high temps increase evapotranspiration.";
      case "cool_spell":
        return "<b>Action:</b> Sow during warmer hours; monitor stress.<br><b>Why:</b> Cool spell may slow germination.";
      case "dry_spell":
        return "<b>Action:</b> Plan irrigation schedule.<br><b>Why:</b> Several dry days ahead will lower soil moisture.";
      default:
        return "<b>Action:</b> No special action needed.<br><b>Why:</b> Weather risk is low.";
    }
  }

  function riskLevelNum(label){ return label==="High"?2 : label==="Moderate"?1 : 0; }

  const mm72  = +((data.rain_mm_72h||0));
  const peakP = +((data.precip_prob_peak_72h||0));
  const gate72 = riskLevelNum(classifyRisk(mm72, peakP).label);

  const next3 = (rows||[]).slice(0,3);
  const maxRisk3 = Math.max(...next3.map(d=>riskLevelNum(classifyRisk(+d.rain_mm||0, +d.prob_max||0).label)), 0);
  const rainSum3 = next3.reduce((s,d)=>s+(+d.rain_mm||0),0);

  let wetSpell = false, streak = 0;
  (rows||[]).slice(0,7).forEach(d=>{
    if ((+d.rain_mm||0) >= 5) { streak++; if (streak>=3) wetSpell=true; }
    else streak=0;
  });

  function harmonizeFlag(flag){
    if (gate72===0 && maxRisk3===0){
      if (flag==="heavy_rain_soon") return {flag:"default", downgradedFrom:"heavy_rain_soon"};
      if (flag==="wet_spell")       return {flag:"default", downgradedFrom:"wet_spell"};
    }
    if (flag==="wet_spell" && !wetSpell) { return {flag:"default", downgradedFrom:"wet_spell"}; }
    if (flag==="heavy_rain_soon" && rainSum3 < 8) { return {flag:"default", downgradedFrom:"heavy_rain_soon"}; }
    if (flag==="flooding" && !rows.some(d=>(+d.rain_mm||0) >= 30)){ return {flag:"default", downgradedFrom:"flooding"}; }
    return {flag, downgradedFrom:null};
  }

  if(!advs.length){
    adviceWrap.innerHTML = `<div class="recCard safe"><div class="recIcon">🌱</div><div class="recText">No recommendations available.</div></div>`;
  } else {
    adviceWrap.innerHTML = advs.map((t,i)=>{
      const raw = flags[i] || inferFlag(t);
      const {flag:f, downgradedFrom} = harmonizeFlag(raw);
      const ic  = iconMap[f] || iconMap.default;
      const txt = beautify(t, f, downgradedFrom);
      const horizon = (i===0 ? "h72" : "d7");
      const badge   = horizon==="h72"
                        ? `<span class="recBadge h72">Next 72h</span>`
                        : `<span class="recBadge d7">Next 7 days</span>`;
      return `<div class="recCard ${f} ${horizon==='h72'?'short':'mid'}">
                <div class="recHead">${badge}</div>
                <div class="recIcon">${ic}</div>
                <div class="recText">${txt}</div>
              </div>`;
    }).join("");
  }
})();
