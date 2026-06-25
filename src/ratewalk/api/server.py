"""RateWalk API + minimal web UI.

    python -m ratewalk.api.server          # serves http://127.0.0.1:8780

Endpoints
  GET /api/health            liveness
  GET /api/config            the active config (as JSON)
  POST /api/run              run the pipeline (optional JSON config override) and
                             return the full report
  GET /                      a single-page UI that renders the report (fan
                             chart, distribution + GMM, duration surface,
                             transition heatmap, sensitivity bands)

This is a scaffold: it serves the real report the engine produces. The richer
React/Plotly front end described in DESIGN.md consumes these same endpoints.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from .. import config as cfgmod
from .. import obs
from ..cli import run_pipeline

try:
    from fastapi import FastAPI
    from fastapi.responses import HTMLResponse
    from pydantic import BaseModel
except Exception as exc:  # pragma: no cover
    raise SystemExit("API extras not installed. pip install -e '.[api]'") from exc


class RunRequest(BaseModel):
    config_path: Optional[str] = None
    country: Optional[str] = None
    n_paths: Optional[int] = None
    source: Optional[str] = None        # auto | fred | synthetic


def create_app() -> "FastAPI":
    app = FastAPI(title="RateWalk", version="0.1.0")

    @app.get("/api/health")
    def health():
        return {"status": "ok", "service": "ratewalk"}

    @app.get("/api/config")
    def get_config():
        c = cfgmod.load()
        from dataclasses import asdict
        return {"config_hash": c.content_hash(), "config": asdict(c)}

    @app.post("/api/run")
    def run(req: RunRequest):
        import dataclasses
        c = cfgmod.load(Path(req.config_path) if req.config_path else None)
        if req.country:
            c = dataclasses.replace(c, country=req.country)
        if req.source:
            c = dataclasses.replace(c, data=dataclasses.replace(c.data, source=req.source))
        if req.n_paths:
            c = dataclasses.replace(c, sim=dataclasses.replace(c.sim, n_paths=int(req.n_paths)))
        obs.configure(log_dir=Path("runs") / "logs")
        return run_pipeline(c)

    @app.get("/", response_class=HTMLResponse)
    def index():
        return _INDEX_HTML

    return app


_INDEX_HTML = """<!doctype html><html><head><meta charset=utf-8>
<title>RateWalk</title>
<script src="https://cdn.plot.ly/plotly-2.32.0.min.js" defer></script>
<style>
 body{background:#0b0f14;color:#e6edf3;font-family:ui-monospace,Menlo,monospace;margin:0;padding:24px}
 h1{color:#e3b341;margin:0 0 2px;font-size:22px} .sub{color:#7d8590;margin-bottom:14px;font-size:13px}
 button{background:#e3b341;color:#0b0f14;border:0;padding:7px 14px;border-radius:6px;cursor:pointer;font-weight:700}
 select,input{background:#11161d;color:#e6edf3;border:1px solid #2a3543;border-radius:6px;padding:6px 8px}
 label{color:#7d8590;font-size:12px;margin-right:4px}
 .bar{display:flex;gap:12px;align-items:center;flex-wrap:wrap;margin-bottom:14px}
 .grid{display:grid;grid-template-columns:1fr 1fr;gap:14px}
 .card{background:#11161d;border:1px solid #222b35;border-radius:8px;padding:12px;min-height:60px}
 .card b{font-size:13px} .full{grid-column:1/3}
 .kpi{font-size:30px;color:#e3b341} .kpis{display:flex;gap:26px;flex-wrap:wrap;align-items:baseline}
 .k2{font-size:18px;color:#e6edf3} .lbl{color:#7d8590;font-size:11px}
 pre{white-space:pre-wrap;color:#9aa7b4;font-size:11px;max-height:300px;overflow:auto}
 .err{color:#ff7b72} .muted{color:#7d8590} .good{color:#3fb950} .bad{color:#ff7b72}
</style></head><body>
<h1>RateWalk</h1><div class=sub>Markov-driven fixed-income path simulation, risk &amp; hedging</div>
<div class=bar>
 <label>country</label><select id=country>
   <option value=US>US</option><option value=GB>GB</option><option value=DE>DE</option>
   <option value=JP>JP</option><option value=CA>CA</option></select>
 <label>paths</label><input id=paths type=number value=4000 min=500 max=20000 step=500 style=width:90px>
 <label>data</label><select id=source><option value=auto>auto</option><option value=fred>fred</option><option value=synthetic>synthetic</option></select>
 <button onclick="go()">Run</button>
 <span id=stat class=sub></span>
</div>
<div class=card style=margin-bottom:14px><div class=kpis id=kpi><span class=muted>running the pipeline (~10s)...</span></div></div>
<div class=grid>
 <div class=card><b>Wealth fan chart (p5-p95 over horizon)</b><div id=fan style=height:300px></div></div>
 <div class=card><b>Short-rate fan chart</b><div id=ratefan style=height:300px></div></div>
 <div class=card><b>Annualized return distribution + GMM</b><div id=dist style=height:300px></div></div>
 <div class=card><b>Transition matrix P(next | current)</b><div id=heat style=height:300px></div></div>
 <div class=card><b>Duration objective surface</b><div id=dur style=height:300px></div></div>
 <div class=card><b>Transition-probability sensitivity (mean return band)</b><div id=sens style=height:300px></div></div>
</div>
<div class=card style=margin-top:14px><b>Full report (JSON)</b><pre id=raw class=muted>waiting...</pre></div>
<script>
const AU='#e3b341', BG='#11161d', MUT='#9aa7b4';
function plot(id,data,layout){
 if(window.Plotly){Plotly.newPlot(id,data,Object.assign({paper_bgcolor:BG,plot_bgcolor:BG,font:{color:MUT,size:11},margin:{t:14,r:10,b:40,l:48},showlegend:false},layout),{displayModeBar:false});}
 else{document.getElementById(id).innerHTML='<span class=muted>(charts need the Plotly CDN; data is in the JSON below)</span>';}
}
function band(t,lo,hi,color){return [
 {x:t,y:hi,mode:'lines',line:{width:0}},
 {x:t,y:lo,mode:'lines',line:{width:0},fill:'tonexty',fillcolor:color}];}
async function go(){
 const stat=document.getElementById('stat'); stat.innerText='running (~10s)...';
 const body=JSON.stringify({country:country.value,n_paths:+paths.value,source:source.value});
 try{
  const r=await fetch('/api/run',{method:'POST',headers:{'Content-Type':'application/json'},body});
  if(!r.ok) throw new Error('HTTP '+r.status);
  const d=await r.json();
  const beats=d.markov.beats_baseline;
  stat.innerHTML='done &middot; '+d.country+' &middot; '+d.data_source+' data &middot; config '+d.config_hash+
    ' &middot; chain '+(beats?'<span class=good>beats baseline</span>':'<span class=bad>below baseline</span>');
  const h=d.headline.annualized_return_pct, t=d.risk.VaR_CVaR['95'];
  document.getElementById('kpi').innerHTML=
    `<div><div class=kpi>${h.p50}%</div><div class=lbl>median annual return (${d.headline.horizon_years}y)</div></div>`+
    `<div><div class=k2>${h.p5}% / ${h.p95}%</div><div class=lbl>p5 / p95</div></div>`+
    `<div><div class=k2>${(t.VaR*100).toFixed(2)}%</div><div class=lbl>VaR95</div></div>`+
    `<div><div class=k2>${(t.CVaR*100).toFixed(2)}%</div><div class=lbl>CVaR95</div></div>`+
    `<div><div class=k2>${d.duration_grid.best.duration}y</div><div class=lbl>best duration</div></div>`+
    `<div><div class=k2>${d.start_short_rate}%</div><div class=lbl>start short rate</div></div>`;
  // wealth fan
  const f=d.viz.fan_chart, T=f.t_years;
  plot('fan',[...band(T,f.wealth.p5,f.wealth.p95,'rgba(227,179,65,0.15)'),
              ...band(T,f.wealth.p25,f.wealth.p75,'rgba(227,179,65,0.30)'),
              {x:T,y:f.wealth.p50,mode:'lines',line:{color:AU,width:2}}],
       {xaxis:{title:'years'},yaxis:{title:'wealth'}});
  plot('ratefan',[...band(T,f.rate.p5,f.rate.p95,'rgba(88,166,255,0.15)'),
              ...band(T,f.rate.p25,f.rate.p75,'rgba(88,166,255,0.30)'),
              {x:T,y:f.rate.p50,mode:'lines',line:{color:'#58a6ff',width:2}}],
       {xaxis:{title:'years'},yaxis:{title:'short rate %'}});
  // distribution + GMM
  const hh=d.viz.return_histogram, traces=[{x:hh.centers,y:hh.counts,type:'bar',marker:{color:'rgba(227,179,65,0.5)'}}];
  if(d.viz.gmm_density){const g=d.viz.gmm_density,sc=Math.max(...hh.counts)/Math.max(...g.density);
    traces.push({x:g.x,y:g.density.map(v=>v*sc),mode:'lines',line:{color:'#ff7b72',width:2},yaxis:'y'});}
  plot('dist',traces,{xaxis:{title:'annual return %'},yaxis:{title:'count'}});
  // transition heatmap
  const tm=d.viz.transition_matrix;
  plot('heat',[{z:tm.P,x:tm.labels,y:tm.labels,type:'heatmap',colorscale:'YlOrBr',
    text:tm.P.map(r=>r.map(v=>v.toFixed(2))),texttemplate:'%{text}',textfont:{size:9}}],
   {xaxis:{title:'next'},yaxis:{title:'current',autorange:'reversed'}});
  // duration surface
  const s=d.duration_grid.surface;
  plot('dur',[{x:s.map(p=>p.duration),y:s.map(p=>p.objective),type:'scatter',mode:'lines+markers',line:{color:AU}}],
   {xaxis:{title:'duration (y)'},yaxis:{title:d.duration_grid.objective}});
  // sensitivity band
  const se=d.transition_sensitivity.mean_return;
  plot('sens',[{x:['p5','mean','p95'],y:[se.p5*100,se.mean*100,se.p95*100],type:'bar',marker:{color:AU}}],
   {xaxis:{title:'Dirichlet draws'},yaxis:{title:'mean ann return %'}});
  document.getElementById('raw').className=''; document.getElementById('raw').innerText=JSON.stringify(d,null,2);
 }catch(e){
  stat.innerHTML='<span class=err>error: '+e.message+'</span>';
  document.getElementById('kpi').innerHTML='<span class=err>run failed: '+e.message+'</span>';
 }
}
window.addEventListener('load',go);
</script></body></html>"""


def main():
    import uvicorn
    obs.configure(log_dir=Path("runs") / "logs")
    uvicorn.run(create_app(), host="127.0.0.1", port=8780)


if __name__ == "__main__":
    main()
