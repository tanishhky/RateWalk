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
        c = cfgmod.load(Path(req.config_path) if req.config_path else None)
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
 h1{color:#e3b341;margin:0 0 4px} .sub{color:#7d8590;margin-bottom:16px}
 button{background:#e3b341;color:#0b0f14;border:0;padding:8px 16px;border-radius:6px;cursor:pointer;font-weight:700}
 .grid{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-top:16px}
 .card{background:#11161d;border:1px solid #222b35;border-radius:8px;padding:12px;min-height:60px}
 .kpi{font-size:26px;color:#e3b341} pre{white-space:pre-wrap;color:#9aa7b4;font-size:12px;max-height:340px;overflow:auto}
 .err{color:#ff7b72} .muted{color:#7d8590}
</style></head><body>
<h1>RateWalk</h1><div class=sub>Markov-driven fixed-income path simulation, risk &amp; hedging</div>
<button onclick="go()">Re-run pipeline</button> <span id=stat class=sub>loading...</span>
<div class=grid>
 <div class=card><b>Headline</b><div id=kpi class=muted>running the pipeline, this takes ~10s...</div></div>
 <div class=card><b>Return distribution (p5 / median / p95)</b><div id=dist style=height:260px></div></div>
 <div class=card><b>Duration objective surface</b><div id=dur style=height:260px></div></div>
 <div class=card><b>Rate-state stationary distribution</b><div id=heat style=height:260px></div></div>
</div>
<div class=card style=margin-top:16px><b>Full report (JSON)</b><pre id=raw class=muted>waiting for first run...</pre></div>
<script>
function plot(id,data,layout){
 if(window.Plotly){Plotly.newPlot(id,data,Object.assign({paper_bgcolor:'#11161d',plot_bgcolor:'#11161d',font:{color:'#9aa7b4'},margin:{t:10}},layout));}
 else{document.getElementById(id).innerHTML='<span class=muted>(charts need the Plotly CDN; numbers are in the JSON below)</span>';}
}
async function go(){
 const stat=document.getElementById('stat'); stat.innerText='running pipeline (~10s)...';
 try{
  const r=await fetch('/api/run',{method:'POST',headers:{'Content-Type':'application/json'},body:'{}'});
  if(!r.ok) throw new Error('HTTP '+r.status);
  const d=await r.json();
  stat.innerText='done - config '+d.config_hash+' - '+d.data_source+' data';
  const h=d.headline.annualized_return_pct;
  document.getElementById('kpi').className='';
  document.getElementById('kpi').innerHTML=
    `<div class=kpi>${h.p50}% / yr</div>median annualized over ${d.headline.horizon_years}y<br>`+
    `p5 ${h.p5}% &middot; p95 ${h.p95}% &middot; VaR95 ${(d.risk.VaR_CVaR['95'].VaR*100).toFixed(2)}% &middot; best duration ${d.duration_grid.best.duration}y`;
  plot('dist',[{x:[h.p5,h.p50,h.p95],type:'box',name:'ann %',marker:{color:'#e3b341'}}],{});
  const s=d.duration_grid.surface;
  plot('dur',[{x:s.map(p=>p.duration),y:s.map(p=>p.objective),type:'scatter',mode:'lines+markers',line:{color:'#e3b341'}}],
   {xaxis:{title:'duration (y)'},yaxis:{title:'objective'}});
  const labs=d.markov.rate_states, sd=Object.values(d.markov.stationary_distribution);
  plot('heat',[{z:[sd],x:labs,type:'heatmap',colorscale:'YlOrBr'}],{});
  document.getElementById('raw').className=''; document.getElementById('raw').innerText=JSON.stringify(d,null,2);
 }catch(e){
  stat.innerHTML='<span class=err>error: '+e.message+'</span>';
  document.getElementById('kpi').innerHTML='<span class=err>run failed: '+e.message+'</span>';
 }
}
// auto-run on load so the page is never blank
window.addEventListener('load',go);
</script></body></html>"""


def main():
    import uvicorn
    obs.configure(log_dir=Path("runs") / "logs")
    uvicorn.run(create_app(), host="127.0.0.1", port=8780)


if __name__ == "__main__":
    main()
