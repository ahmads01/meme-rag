# ================================================
# app.py — Web UI Only
# All RAG logic lives in meme_rag.py
# ================================================

import os
from flask import Flask, request, jsonify, send_file
from meme_rag import load_or_create_metadata, build_index, find_memes, MEMES_FOLDER

app = Flask(__name__)

# built once when server starts
index     = None
meme_list = None


# ================================================
# ROUTES
# ================================================

@app.route('/')
def home():
    return HTML_PAGE


@app.route('/image/<path:filename>')
def serve_image(filename):
    """Serves the actual image file to the browser."""
    filepath = os.path.join(os.path.abspath(MEMES_FOLDER), filename)
    return send_file(filepath)


@app.route('/search', methods=['POST'])
def search():
    """Receives query from browser, returns top 3 memes as JSON."""
    query = request.json.get('query', '').strip()
    if not query:
        return jsonify({"error": "Empty query"}), 400
    try:
        results = find_memes(query, index, meme_list)
        return jsonify({"results": results, "query": query})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/stats')
def stats():
    """Returns total meme count and category breakdown for the header."""
    categories = {}
    for m in meme_list:
        c = m.get('category', 'other')
        categories[c] = categories.get(c, 0) + 1
    return jsonify({"total": len(meme_list), "categories": categories})


# ================================================
# UI
# ================================================

HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>MemeRAG</title>
<style>
  *{margin:0;padding:0;box-sizing:border-box}
  body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:#0f0f0f;color:#f0f0f0;min-height:100vh}
  header{padding:24px 40px;border-bottom:1px solid #1e1e1e;display:flex;align-items:center;justify-content:space-between}
  header h1{font-size:20px;font-weight:600;letter-spacing:-0.3px}
  header h1 span{color:#a78bfa}
  #stats-bar{font-size:12px;color:#444}
  .search-wrap{max-width:640px;margin:56px auto 32px;padding:0 20px}
  .search-wrap p{font-size:13px;color:#444;margin-bottom:14px;line-height:1.6}
  .search-row{display:flex;gap:8px}
  input{flex:1;padding:13px 16px;background:#151515;border:1px solid #2a2a2a;border-radius:10px;color:#f0f0f0;font-size:14px;outline:none;transition:border-color .2s}
  input:focus{border-color:#a78bfa}
  input::placeholder{color:#333}
  button#btn{padding:13px 22px;background:#a78bfa;color:#0f0f0f;border:none;border-radius:10px;font-size:14px;font-weight:600;cursor:pointer;transition:background .15s,transform .1s;white-space:nowrap}
  button#btn:hover{background:#c4b5fd}
  button#btn:active{transform:scale(.98)}
  button#btn:disabled{background:#222;color:#555;cursor:not-allowed}
  .chips{display:flex;flex-wrap:wrap;gap:6px;margin-top:12px}
  .chip{padding:5px 13px;background:#151515;border:1px solid #222;border-radius:20px;font-size:12px;color:#666;cursor:pointer;transition:all .15s}
  .chip:hover{border-color:#a78bfa;color:#a78bfa}
  #status{text-align:center;font-size:13px;color:#444;margin:16px 0;min-height:18px}
  .spin{display:inline-block;width:12px;height:12px;border:2px solid #2a2a2a;border-top-color:#a78bfa;border-radius:50%;animation:spin .7s linear infinite;margin-right:6px;vertical-align:middle}
  @keyframes spin{to{transform:rotate(360deg)}}
  #grid{max-width:1080px;margin:0 auto 60px;padding:0 20px;display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:16px}
  .card{background:#141414;border:1px solid #1e1e1e;border-radius:12px;overflow:hidden;transition:transform .2s,border-color .2s;animation:up .25s ease}
  @keyframes up{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:translateY(0)}}
  .card:hover{transform:translateY(-2px);border-color:#2a2a2a}
  .card.best{border-color:#a78bfa}
  .img-box{width:100%;aspect-ratio:1;background:#0a0a0a;display:flex;align-items:center;justify-content:center;overflow:hidden}
  .img-box img{width:100%;height:100%;object-fit:contain}
  .info{padding:14px}
  .row1{display:flex;align-items:flex-start;justify-content:space-between;gap:8px;margin-bottom:8px}
  .title{font-size:13px;font-weight:600;line-height:1.4;color:#f0f0f0}
  .best-pill{font-size:10px;font-weight:600;background:#a78bfa18;color:#a78bfa;border:1px solid #a78bfa33;padding:2px 8px;border-radius:20px;white-space:nowrap;flex-shrink:0}
  .tags{display:flex;flex-wrap:wrap;gap:5px;margin-bottom:8px}
  .tag{font-size:11px;padding:2px 8px;border-radius:20px;border:1px solid #222;color:#555}
  .tag.cat{color:#a78bfa;border-color:#a78bfa22;background:#a78bfa0a}
  .meta{display:flex;align-items:center;justify-content:space-between;font-size:11px;color:#444;margin-bottom:8px}
  .stars{color:#f59e0b}
  .path{font-size:10px;color:#333;font-family:monospace;word-break:break-all;background:#0a0a0a;padding:5px 7px;border-radius:5px;line-height:1.5}
  .empty{text-align:center;color:#333;font-size:13px;padding:60px;grid-column:1/-1}
</style>
</head>
<body>

<header>
  <h1>Meme<span>RAG</span></h1>
  <div id="stats-bar">loading...</div>
</header>

<div class="search-wrap">
  <p>Describe the meme you want in plain English. AI finds the best match.</p>
  <div class="search-row">
    <input type="text" id="q" placeholder="e.g. dark meme about Mondays...">
    <button id="btn" onclick="search()">Search</button>
  </div>
  <div class="chips">
    <span class="chip" onclick="go('dark humor')">dark humor</span>
    <span class="chip" onclick="go('dad joke')">dad joke</span>
    <span class="chip" onclick="go('relatable work')">relatable work</span>
    <span class="chip" onclick="go('funny')">funny</span>
    <span class="chip" onclick="go('programming humor')">programming</span>
    <span class="chip" onclick="go('funny food meme')">food</span>
  </div>
</div>

<div id="status"></div>
<div id="grid"></div>

<script>
fetch('/stats').then(r=>r.json()).then(d=>{
  document.getElementById('stats-bar').textContent=
    `${d.total} memes · ${Object.keys(d.categories).length} categories`;
});

document.getElementById('q').addEventListener('keydown',e=>{
  if(e.key==='Enter') search();
});

function go(t){ document.getElementById('q').value=t; search(); }

function stars(n){
  return '★'.repeat(Math.round(n/2))+'☆'.repeat(5-Math.round(n/2));
}

async function search(){
  const query=document.getElementById('q').value.trim();
  if(!query) return;

  const btn=document.getElementById('btn');
  const status=document.getElementById('status');
  const grid=document.getElementById('grid');

  btn.disabled=true; btn.textContent='Searching...';
  status.innerHTML='<span class="spin"></span>Finding and ranking with AI...';
  grid.innerHTML='';

  try{
    const res=await fetch('/search',{
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({query})
    });
    const data=await res.json();

    if(data.error){ status.textContent='Error: '+data.error; return; }

    status.textContent=`${data.results.length} results for "${data.query}"`;

    if(!data.results.length){
      grid.innerHTML='<div class="empty">No memes found. Try different words.</div>';
      return;
    }

    grid.innerHTML=data.results.map((r,i)=>`
      <div class="card ${i===0?'best':''}">
        <div class="img-box">
          <img src="/image/${encodeURIComponent(r.filename)}"
               alt="${r.title}" loading="lazy"
               onerror="this.parentElement.innerHTML='<div style=color:#333;font-size:12px;padding:20px>Image not found</div>'">
        </div>
        <div class="info">
          <div class="row1">
            <div class="title">${r.title}</div>
            ${i===0?'<span class="best-pill">Best Match</span>':''}
          </div>
          <div class="tags">
            <span class="tag cat">${r.category}</span>
            ${r.keywords.slice(0,3).map(k=>`<span class="tag">${k}</span>`).join('')}
          </div>
          <div class="meta">
            <span class="stars">${stars(r.funniness)} <span style="color:#444">${r.funniness}/10</span></span>
            <span>Match: ${r.score}%</span>
          </div>
          <div class="path">${r.path}</div>
        </div>
      </div>
    `).join('');

  }catch(e){
    status.textContent='Something went wrong. Is the server running?';
  }finally{
    btn.disabled=false; btn.textContent='Search';
  }
}
</script>
</body>
</html>"""


# ================================================
# START
# ================================================

if __name__ == '__main__':
    metadata          = load_or_create_metadata()
    index, meme_list  = build_index(metadata)
    print("Running at http://localhost:5000\n")
    app.run(debug=False, port=5000)