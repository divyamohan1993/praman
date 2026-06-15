"""Self-contained enterprise landing + live verifier console served at /.

No external resources (fonts/scripts/images): CSP-clean, air-gap-consistent, fast.
The inline <script> carries a per-request nonce (__NONCE__ placeholder, replaced by the
route). Dynamic text is rendered with textContent only (no innerHTML on user data) -> XSS-safe.
"""

PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="description" content="PRAMAN: on-prem, CPU-only, Indic-first grounded-claim verifier with a distribution-free risk bound and a regulator-ready audit trail.">
<title>PRAMAN — proof for what AI says</title>
<style>
:root{
  --ink:#070a10; --panel:#0d131d; --panel-2:#0a0f18; --line:#1b2536; --line-2:#26344a;
  --txt:#c4d0e0; --txt-dim:#7c8aa0; --txt-bright:#eef4fb; --accent:#5fb6e6; --accent-2:#2b6f93;
  --accept:#3ad29a; --escalate:#f3b24c; --reject:#f0596b;
  --serif:"Charter","Iowan Old Style","Palatino Linotype",Georgia,"Times New Roman",serif;
  --mono:ui-monospace,"SF Mono","Cascadia Code","JetBrains Mono",Menlo,Consolas,monospace;
  --sans:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,system-ui,sans-serif;
}
*{box-sizing:border-box}
html{scroll-behavior:smooth}
body{
  margin:0;background:var(--ink);color:var(--txt);font-family:var(--sans);font-size:15px;line-height:1.6;
  -webkit-font-smoothing:antialiased;
  background-image:
    linear-gradient(rgba(95,182,230,.035) 1px,transparent 1px),
    linear-gradient(90deg,rgba(95,182,230,.035) 1px,transparent 1px),
    radial-gradient(1200px 600px at 70% -10%,rgba(95,182,230,.10),transparent 60%);
  background-size:46px 46px,46px 46px,100% 100%;
}
a{color:var(--accent);text-decoration:none}a:hover{text-decoration:underline}
.mono{font-family:var(--mono)}
.wrap{max-width:1120px;margin:0 auto;padding:0 24px}
.eyebrow{font-family:var(--mono);font-size:11px;letter-spacing:.22em;text-transform:uppercase;color:var(--accent)}

/* header */
header{position:sticky;top:0;z-index:20;backdrop-filter:blur(10px);
  background:linear-gradient(180deg,rgba(7,10,16,.92),rgba(7,10,16,.66));border-bottom:1px solid var(--line)}
.hbar{display:flex;align-items:center;justify-content:space-between;height:62px}
.brand{font-family:var(--serif);font-size:23px;letter-spacing:.02em;color:var(--txt-bright);display:flex;gap:.5rem;align-items:baseline}
.brand .dev{font-size:18px;color:var(--txt-dim)}
.nav{display:flex;gap:26px;align-items:center;font-family:var(--mono);font-size:12.5px}
.nav a{color:var(--txt-dim)}.nav a:hover{color:var(--txt-bright);text-decoration:none}
.dot{display:inline-flex;align-items:center;gap:7px;font-family:var(--mono);font-size:11.5px;color:var(--accept)}
.dot i{width:8px;height:8px;border-radius:50%;background:var(--accept);box-shadow:0 0 10px var(--accept);animation:pulse 2.4s ease-in-out infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.35}}

/* hero */
.hero{padding:74px 0 30px}
.hero h1{font-family:var(--serif);font-weight:600;font-size:clamp(38px,6vw,64px);line-height:1.04;letter-spacing:-.01em;color:var(--txt-bright);margin:18px 0 0;max-width:16ch}
.hero h1 em{font-style:italic;color:var(--accent)}
.hero .lead{font-family:var(--serif);font-size:clamp(17px,2.2vw,21px);color:#b7c4d6;max-width:62ch;margin:22px 0 0}
.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin:38px 0 4px}
.stat{border:1px solid var(--line);border-radius:12px;padding:16px 16px 14px;background:linear-gradient(180deg,var(--panel),var(--panel-2))}
.stat .k{font-family:var(--mono);font-size:23px;color:var(--txt-bright);letter-spacing:-.01em}
.stat .l{font-family:var(--mono);font-size:11px;letter-spacing:.04em;color:var(--txt-dim);margin-top:5px;text-transform:uppercase}
.stat .k small{font-size:13px;color:var(--accept)}

/* console */
.console{margin:44px 0 10px;border:1px solid var(--line-2);border-radius:16px;overflow:hidden;
  background:linear-gradient(180deg,var(--panel),var(--panel-2));box-shadow:0 30px 80px -40px rgba(0,0,0,.8)}
.console .bar{display:flex;align-items:center;gap:10px;padding:13px 18px;border-bottom:1px solid var(--line);font-family:var(--mono);font-size:12px;color:var(--txt-dim)}
.console .bar b{color:var(--txt-bright);font-weight:600}
.tl{display:flex;gap:6px}.tl i{width:10px;height:10px;border-radius:50%;background:#23303f}
.grid{display:grid;grid-template-columns:1fr 1fr;min-height:520px}
.pane{padding:22px}
.pane.in{border-right:1px solid var(--line)}
.lbl{font-family:var(--mono);font-size:11px;letter-spacing:.08em;text-transform:uppercase;color:var(--txt-dim);display:block;margin:0 0 7px}
.presets{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:18px}
.chip{font-family:var(--mono);font-size:12px;color:var(--txt);background:#0e1622;border:1px solid var(--line-2);
  border-radius:999px;padding:6px 12px;cursor:pointer;transition:.16s}
.chip:hover{border-color:var(--accent);color:var(--txt-bright)}
.chip b{color:var(--accent);font-weight:600}
textarea,select{width:100%;background:var(--panel-2);border:1px solid var(--line-2);border-radius:10px;color:var(--txt-bright);
  font-family:var(--mono);font-size:13px;padding:11px 12px;resize:vertical;transition:.16s}
textarea:focus,select:focus{outline:none;border-color:var(--accent);box-shadow:0 0 0 3px rgba(95,182,230,.14)}
.field{margin-bottom:16px}
.controls{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-bottom:18px}
.seg{display:flex;border:1px solid var(--line-2);border-radius:10px;overflow:hidden}
.seg button{flex:1;background:var(--panel-2);border:0;color:var(--txt-dim);font-family:var(--mono);font-size:13px;padding:9px 0;cursor:pointer;transition:.14s}
.seg button[aria-pressed=true]{background:var(--accent-2);color:var(--txt-bright)}
.run{width:100%;border:0;border-radius:11px;cursor:pointer;font-family:var(--mono);font-size:14px;letter-spacing:.03em;
  padding:14px;color:#04222e;background:linear-gradient(180deg,#7fc8ee,#4ea6db);font-weight:700;transition:.16s;box-shadow:0 8px 24px -10px rgba(95,182,230,.6)}
.run:hover{filter:brightness(1.06)}.run:disabled{opacity:.55;cursor:wait}

/* output pane */
.pane.out{position:relative;background:
  radial-gradient(600px 300px at 80% -20%,rgba(95,182,230,.06),transparent 60%),var(--panel-2)}
.empty{height:100%;display:flex;flex-direction:column;align-items:center;justify-content:center;text-align:center;color:var(--txt-dim);font-family:var(--mono);font-size:13px;gap:12px}
.empty svg{opacity:.4}
.scan{position:absolute;inset:0;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:16px;font-family:var(--mono);font-size:12.5px;color:var(--accent)}
.scan .ring{width:40px;height:40px;border:3px solid rgba(95,182,230,.2);border-top-color:var(--accent);border-radius:50%;animation:spin .9s linear infinite}
@keyframes spin{to{transform:rotate(360deg)}}
.verdict{margin-bottom:18px}
.badge{display:inline-flex;align-items:center;gap:9px;font-family:var(--mono);font-size:13px;letter-spacing:.06em;text-transform:uppercase;
  padding:9px 15px;border-radius:10px;font-weight:700}
.badge::before{content:"";width:9px;height:9px;border-radius:50%;background:currentColor;box-shadow:0 0 10px currentColor}
.b-accept{color:var(--accept);background:rgba(58,210,154,.10);border:1px solid rgba(58,210,154,.3)}
.b-escalate{color:var(--escalate);background:rgba(243,178,76,.10);border:1px solid rgba(243,178,76,.3)}
.b-reject{color:var(--reject);background:rgba(240,89,107,.10);border:1px solid rgba(240,89,107,.3)}
.verdict .sub{font-family:var(--mono);font-size:11.5px;color:var(--txt-dim);margin-top:9px}
.claim{border:1px solid var(--line);border-radius:12px;padding:14px;margin-bottom:11px;background:var(--panel)}
.claim .ct{font-size:14px;color:var(--txt-bright);line-height:1.45}
.claim .meter{height:8px;border-radius:6px;background:#0c1420;margin:12px 0 6px;overflow:hidden;border:1px solid var(--line)}
.claim .meter i{display:block;height:100%;width:0;border-radius:6px;transition:width .9s cubic-bezier(.2,.7,.2,1)}
.claim .row{display:flex;justify-content:space-between;align-items:center;font-family:var(--mono);font-size:11.5px;color:var(--txt-dim)}
.tag{font-family:var(--mono);font-size:10.5px;letter-spacing:.05em;text-transform:uppercase;padding:3px 8px;border-radius:6px;font-weight:700}
.t-accept{color:var(--accept);background:rgba(58,210,154,.12)}
.t-escalate{color:var(--escalate);background:rgba(243,178,76,.12)}
.t-reject{color:var(--reject);background:rgba(240,89,107,.12)}
.span{font-family:var(--mono);font-size:11.5px;color:var(--txt-dim);margin-top:9px;padding-left:10px;border-left:2px solid var(--line-2)}
details.audit{margin-top:14px;border:1px solid var(--line);border-radius:12px;background:var(--panel)}
details.audit summary{cursor:pointer;font-family:var(--mono);font-size:12px;color:var(--txt);padding:12px 14px;list-style:none}
details.audit summary::-webkit-details-marker{display:none}
details.audit summary::before{content:"▸ ";color:var(--accent)}
details.audit[open] summary::before{content:"▾ "}
.kv{padding:0 14px 14px}.kv div{display:flex;gap:12px;font-family:var(--mono);font-size:11.5px;padding:5px 0;border-top:1px solid var(--line)}
.kv span:first-child{color:var(--txt-dim);min-width:118px}.kv span:last-child{color:var(--txt-bright);word-break:break-all}
.err{color:var(--reject);font-family:var(--mono);font-size:12.5px;border:1px solid rgba(240,89,107,.3);border-radius:10px;padding:14px;background:rgba(240,89,107,.06)}

/* sections */
section.band{padding:70px 0;border-top:1px solid var(--line)}
section.band h2{font-family:var(--serif);font-size:clamp(26px,3.6vw,38px);color:var(--txt-bright);margin:10px 0 0;font-weight:600}
.steps{display:grid;grid-template-columns:repeat(4,1fr);gap:16px;margin-top:34px}
.step{border:1px solid var(--line);border-radius:12px;padding:18px;background:var(--panel)}
.step .n{font-family:var(--mono);font-size:12px;color:var(--accent)}
.step h3{font-family:var(--serif);font-size:18px;color:var(--txt-bright);margin:8px 0 6px;font-weight:600}
.step p{font-size:13.5px;color:var(--txt-dim);margin:0}
.scope{border:1px solid var(--line-2);border-left:3px solid var(--escalate);border-radius:12px;padding:22px 24px;background:var(--panel);margin-top:30px}
.scope p{margin:0;font-family:var(--serif);font-size:16px;color:#c8d4e4}
.scope b{color:var(--txt-bright)}

footer{border-top:1px solid var(--line);padding:40px 0 56px;margin-top:30px;color:var(--txt-dim);font-family:var(--mono);font-size:12px}
.fgrid{display:flex;justify-content:space-between;flex-wrap:wrap;gap:20px}
.fgrid a{color:var(--txt-dim)}.fgrid a:hover{color:var(--txt-bright);text-decoration:none}

[data-reveal]{opacity:0;transform:translateY(14px);animation:rise .7s cubic-bezier(.2,.7,.2,1) forwards}
@keyframes rise{to{opacity:1;transform:none}}
@media (max-width:860px){.grid{grid-template-columns:1fr}.pane.in{border-right:0;border-bottom:1px solid var(--line)}
  .stats,.steps{grid-template-columns:1fr 1fr}.nav{display:none}}
@media (prefers-reduced-motion:reduce){*{animation:none!important;transition:none!important}[data-reveal]{opacity:1;transform:none}}
:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
</style>
</head>
<body>
<header><div class="wrap hbar">
  <div class="brand">PRAMAN <span class="dev">प्रमाण</span></div>
  <nav class="nav">
    <a href="#verify">Try it</a><a href="#how">How it works</a>
    <a href="/docs">API</a><a href="https://github.com/divyamohan1993/praman">Source</a>
    <span class="dot"><i></i>LIVE</span>
  </nav>
</div></header>

<main>
<section class="wrap hero">
  <div class="eyebrow" data-reveal>On-prem · CPU-only · Air-gapped · Indic-first</div>
  <h1 data-reveal style="animation-delay:.05s">Proof for what your AI <em>says</em>.</h1>
  <p class="lead" data-reveal style="animation-delay:.1s">PRAMAN checks a generated output against the evidence it should rest on, and returns, per claim, a calibrated verdict, a <b>distribution-free bound</b> on the rate of auto-approving an ungrounded claim, an accept / escalate / reject decision, and a regulator-ready audit record. It bounds a rate, it is not a per-item certificate; it keeps the human on the decisions that matter.</p>
  <div class="stats" data-reveal style="animation-delay:.16s">
    <div class="stat"><div class="k">&le; &alpha; <small>guaranteed</small></div><div class="l">Missed-approval rate</div></div>
    <div class="stat"><div class="k">0.005<small>/0.038/0.076</small></div><div class="l">Realized FNR @ &alpha;=.01/.05/.10</div></div>
    <div class="stat"><div class="k">100%</div><div class="l">On-prem · no external calls</div></div>
    <div class="stat"><div class="k">1 / claim</div><div class="l">Tamper-evident audit record</div></div>
  </div>
</section>

<section class="wrap" id="verify">
<div class="console" data-reveal>
  <div class="bar"><span class="tl"><i></i><i></i><i></i></span> <b>verifier</b> · live console <span style="margin-left:auto" id="lat"></span></div>
  <div class="grid">
    <div class="pane in">
      <span class="lbl">Presets</span>
      <div class="presets" id="presets"></div>
      <div class="field"><span class="lbl">Generated output</span><textarea id="output" rows="3" placeholder="The claim(s) to verify..."></textarea></div>
      <div class="field"><span class="lbl">Evidence — one passage per line</span><textarea id="evidence" rows="4" placeholder="What the output should be grounded in..."></textarea></div>
      <div class="controls">
        <div><span class="lbl">Risk level &alpha;</span><div class="seg" id="alpha">
          <button data-a="0.01">0.01</button><button data-a="0.05" aria-pressed="true">0.05</button><button data-a="0.10">0.10</button></div></div>
        <div><span class="lbl">Policy class</span><select id="cls">
          <option value="general">general</option><option value="clinical">clinical · high</option>
          <option value="financial">financial · high</option><option value="legal">legal · high</option>
          <option value="bulk">bulk · reversible</option></select></div>
      </div>
      <button class="run" id="run">Verify &#9656;</button>
    </div>
    <div class="pane out" id="out">
      <div class="empty" id="empty">
        <svg width="46" height="46" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.4"><path d="M9 12l2 2 4-4"/><circle cx="12" cy="12" r="9"/></svg>
        <div>Load a preset or enter an output + evidence,<br>then run a verification.</div>
      </div>
    </div>
  </div>
</div>
</section>

<section class="wrap band" id="how">
  <div class="eyebrow">The method</div>
  <h2>A bound you can defend, not a vibe score.</h2>
  <div class="steps">
    <div class="step"><div class="n">01</div><h3>Decompose</h3><p>Split the output into atomic, independently checkable claims.</p></div>
    <div class="step"><div class="n">02</div><h3>Score &amp; calibrate</h3><p>A small CPU model scores each claim against the evidence; scores are calibrated into real probabilities.</p></div>
    <div class="step"><div class="n">03</div><h3>Conformal control</h3><p>Conformal risk control picks the threshold so the missed-approval rate is provably &le; &alpha;.</p></div>
    <div class="step"><div class="n">04</div><h3>Decide &amp; log</h3><p>Accept the safe majority, escalate the uncertain, and write a content-hashed audit record.</p></div>
  </div>
  <div class="scope">
    <p><b>Stated truthfully:</b> PRAMAN gives provably right-sized review and a defensible audit trail. The guarantee is marginal, not a per-item certificate; it is faithfulness to the supplied evidence, not truth in the world; and it does not remove the human on catastrophic or irreversible decisions. This managed-cloud page is a demo; the product runs air-gapped on the operator's own hardware.</p>
  </div>
</section>
</main>

<footer><div class="wrap fgrid">
  <div>PRAMAN · grounded-claim verifier · Apache-2.0 · Divya Mohan (dmj.one)</div>
  <div style="display:flex;gap:22px">
    <a href="/docs">API docs</a><a href="/health">Health</a>
    <a href="https://github.com/divyamohan1993/praman">GitHub</a>
    <a href="https://github.com/divyamohan1993/praman/blob/main/REPORT.md">Technical report</a>
  </div>
</div></footer>

<script nonce="__NONCE__">
(function(){
  "use strict";
  var $=function(s){return document.querySelector(s)};
  var PRESETS=[
    {t:'Clinical · date',o:'The drug was approved in 2019 and cuts mortality by 40%.',
     e:'A regulatory review concluded the agency approved the drug in 2021.',a:'0.05',c:'clinical'},
    {t:'False · landmark',o:'The Eiffel Tower is in Berlin.',
     e:'The Eiffel Tower is a landmark in Paris, France.',a:'0.05',c:'general'},
    {t:'Grounded · capital',o:'Paris is the capital of France.',
     e:'Paris is the capital and most populous city of France.',a:'0.05',c:'general'},
    {t:'Mixed',o:'Water boils at 50 degrees Celsius. Ice is frozen water.',
     e:'At sea-level pressure water boils at 100 degrees Celsius. Ice is the solid state of water.',a:'0.10',c:'bulk'}
  ];
  var pc=$('#presets');
  PRESETS.forEach(function(p){
    var b=document.createElement('button');b.className='chip';
    var s=document.createElement('b');s.textContent=p.t;b.appendChild(s);
    b.addEventListener('click',function(){
      $('#output').value=p.o;$('#evidence').value=p.e;setAlpha(p.a);$('#cls').value=p.c;
    });
    pc.appendChild(b);
  });
  function setAlpha(a){Array.prototype.forEach.call(document.querySelectorAll('#alpha button'),function(x){
    x.setAttribute('aria-pressed', x.getAttribute('data-a')===a?'true':'false');});}
  document.querySelectorAll('#alpha button').forEach(function(x){
    x.addEventListener('click',function(){setAlpha(x.getAttribute('data-a'))});});
  function alpha(){var b=$('#alpha button[aria-pressed=true]');return b?parseFloat(b.getAttribute('data-a')):0.05;}
  function cls(){var v=$('#cls').value;var hi={clinical:1,financial:1,legal:1};
    return {cls:v,sev:hi[v]?'high':'normal'};}
  var DEC={accept:'accept',escalate:'escalate',reject:'reject'};

  function el(tag,cls,txt){var e=document.createElement(tag);if(cls)e.className=cls;if(txt!=null)e.textContent=txt;return e;}

  function render(d,ms){
    var out=$('#out');out.textContent='';
    var od=d.output_decision;
    var v=el('div','verdict');
    var bd=el('div','badge b-'+od, 'output · '+od);
    v.appendChild(bd);
    v.appendChild(el('div','sub','α = '+d.alpha+'  ·  policy '+(d.policy&&d.policy.class||'')+'  ·  '+d.claims.length+' claim(s)  ·  verified in '+ms+' ms'));
    out.appendChild(v);
    d.claims.forEach(function(c,i){
      var card=el('div','claim');card.style.animationDelay=(i*0.05)+'s';card.setAttribute('data-reveal','');
      card.appendChild(el('div','ct',c.text));
      var m=el('div','meter');var bar=el('i');
      var col=c.decision==='accept'?'var(--accept)':c.decision==='reject'?'var(--reject)':'var(--escalate)';
      bar.style.background=col;m.appendChild(bar);card.appendChild(m);
      var row=el('div','row');
      row.appendChild(el('span',null,'P(grounded) = '+Number(c.p_grounded).toFixed(3)));
      row.appendChild(el('span','tag t-'+c.decision,c.decision));
      card.appendChild(row);
      if(c.evidence_span){var sp=el('div','span','matched evidence: '+c.evidence_span);card.appendChild(sp);}
      out.appendChild(card);
      requestAnimationFrame(function(){bar.style.width=(Math.max(2,Math.min(100,c.p_grounded*100)))+'%';});
    });
    // audit
    if(d.audit&&d.audit.length){
      var a=d.audit[0];var det=el('details','audit');var sm=el('summary',null,'Audit record  ·  regulator-ready trail ('+d.audit.length+')');det.appendChild(sm);
      var kv=el('div','kv');
      [['decision',a.decision],['p_grounded',a.p_grounded],['policy.alpha',a.policy&&a.policy.alpha],
       ['policy.class',a.policy&&a.policy.class],['method',a.policy&&a.policy.method],
       ['model_version',a.model_version],['calib_version',a.calib_version],['content_hash',a.content_hash]]
      .forEach(function(p){var r=el('div');r.appendChild(el('span',null,p[0]));r.appendChild(el('span',null,String(p[1])));kv.appendChild(r);});
      det.appendChild(kv);out.appendChild(det);
    }
  }

  function run(){
    var output=$('#output').value.trim();
    var evidence=$('#evidence').value.split('\n').map(function(s){return s.trim()}).filter(Boolean);
    if(!output||!evidence.length){$('#output').focus();return;}
    var btn=$('#run');btn.disabled=true;
    var out=$('#out');out.textContent='';
    var sc=el('div','scan');sc.appendChild(el('div','ring'));sc.appendChild(el('div',null,'measuring groundedness · applying conformal threshold…'));out.appendChild(sc);
    var c=cls();var t0=performance.now();
    fetch('/verify',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({output_text:output,evidence:evidence,alpha:alpha(),policy:{class:c.cls,severity:c.sev}})})
    .then(function(r){return r.json().then(function(j){return {ok:r.ok,j:j}})})
    .then(function(x){
      var ms=Math.round(performance.now()-t0);
      if(!x.ok){throw new Error((x.j&&x.j.detail)||('HTTP '+ms));}
      render(x.j,ms);$('#lat').textContent=ms+' ms';
    })
    .catch(function(e){
      out.textContent='';
      var er=el('div','err');
      er.textContent='Verification failed: '+e.message+'. On a cold start the instrument loads its model on the first call (~20s) — try again.';
      out.appendChild(er);
    })
    .finally(function(){btn.disabled=false;});
  }
  $('#run').addEventListener('click',run);
  // load the first preset so the console is never empty
  pc.firstChild.click();
})();
</script>
</body>
</html>"""
