import json
import os
import re
import webbrowser
from collections import defaultdict


def invert_data(data):
    """
    Inverts the data structure from:
        patent -> descriptions/claims -> article chunks
    to:
        chunk -> top descriptions + top claims across all patents
    """
    chunk_descriptions = defaultdict(list)
    chunk_claims = defaultdict(list)
    chunk_contents = {}

    for patent in data:
        patent_id = patent["patent_id"]

        for desc in patent["descriptions"]:
            for sim in desc["article_similarities"]:
                chunk_name = sim["chunk_name"]
                chunk_contents[chunk_name] = sim["chunk_content"]
                chunk_descriptions[chunk_name].append({
                    "patent_id": patent_id,
                    "marker": desc["marker"],
                    "content": desc["description_content"],
                    "similarity": sim["similarity"],
                })

        for claim in patent["claims"]:
            for sim in claim["article_similarities"]:
                chunk_name = sim["chunk_name"]
                chunk_contents[chunk_name] = sim["chunk_content"]
                chunk_claims[chunk_name].append({
                    "patent_id": patent_id,
                    "marker": claim["marker"],
                    "content": claim["claim_content"],
                    "similarity": sim["similarity"],
                })

    result = {}
    for chunk_name, content in chunk_contents.items():
        top_descs = sorted(chunk_descriptions[chunk_name], key=lambda x: x["similarity"], reverse=True)[:3]
        top_claims = sorted(chunk_claims[chunk_name], key=lambda x: x["similarity"], reverse=True)[:3]
        result[chunk_name] = {
            "chunk_content": content,
            "top_descriptions": top_descs,
            "top_claims": top_claims,
        }

    return result


def group_by_article(inverted_data):
    """
    Groups chunks by article name (strips _chunk_N suffix) and sorts by index.
    Returns: {article_name: [[idx, chunk_name], ...]}
    """
    articles = defaultdict(list)
    for chunk_name in inverted_data:
        match = re.match(r'^(.+)_chunk_(\d+)$', chunk_name)
        if match:
            article_name = match.group(1)
            chunk_idx = int(match.group(2))
            articles[article_name].append([chunk_idx, chunk_name])

    for article_name in articles:
        articles[article_name].sort(key=lambda x: x[0])

    return dict(articles)


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Patent-Article Similarity</title>
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{ font-family: system-ui, -apple-system, sans-serif; background: #f0f2f5; color: #222; }}

    .layout {{ display: flex; height: 100vh; overflow: hidden; }}

    /* Left panel */
    .articles-panel {{
      flex: 1;
      overflow-y: auto;
      padding: 24px;
      border-right: 1px solid #ddd;
      background: white;
    }}

    .panel-title {{
      font-size: 17px;
      font-weight: 700;
      margin-bottom: 20px;
      color: #111;
    }}

    .article-section {{ margin-bottom: 32px; }}

    .article-title {{
      font-size: 11px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: #999;
      margin-bottom: 10px;
      padding-bottom: 6px;
      border-bottom: 2px solid #eee;
    }}
    .article-source-link {{
      font-size: 10px;
      color: #4f6fff;
      text-decoration: none;
      font-weight: 400;
      letter-spacing: 0;
      text-transform: none;
    }}
    .article-source-link:hover {{ text-decoration: underline; }}

    .chunk {{
      padding: 10px 14px;
      margin-bottom: 6px;
      border-radius: 6px;
      border: 2px solid transparent;
      cursor: pointer;
      font-size: 13px;
      line-height: 1.6;
      background: #f7f8fa;
      transition: background 0.12s, border-color 0.12s;
    }}
    .chunk:hover, .chunk.active {{
      background: #eef2ff;
      border-color: #4f6fff;
    }}
    .chunk-label {{
      font-size: 10px;
      font-weight: 700;
      color: #bbb;
      margin-bottom: 4px;
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }}

    /* Right panel */
    .results-panel {{
      width: 400px;
      flex-shrink: 0;
      overflow-y: auto;
      padding: 24px;
      background: #fafbfc;
    }}

    .results-panel .panel-title {{ color: #555; font-size: 14px; }}

    .placeholder {{
      color: #bbb;
      font-size: 13px;
      text-align: center;
      margin-top: 60px;
      line-height: 1.7;
    }}

    .section-label {{
      font-size: 10px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: #aaa;
      margin: 18px 0 8px;
    }}
    .section-label:first-child {{ margin-top: 0; }}

    .result-card {{
      background: white;
      border: 1px solid #e4e7ec;
      border-radius: 8px;
      padding: 12px 14px;
      margin-bottom: 8px;
    }}

    .result-meta {{
      display: flex;
      justify-content: space-between;
      align-items: baseline;
      margin-bottom: 4px;
    }}
    .result-marker {{
      font-size: 12px;
      font-weight: 700;
      color: #4f6fff;
    }}
    .result-patent {{
      font-size: 10px;
      color: #bbb;
    }}
    .result-title {{
      font-size: 11px;
      font-weight: 600;
      color: #666;
      margin-bottom: 2px;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }}
    .result-espacenet {{
      font-size: 10px;
      color: #4f6fff;
      text-decoration: none;
      margin-bottom: 6px;
      display: inline-block;
    }}
    .result-espacenet:hover {{ text-decoration: underline; }}

    .sim-bar {{
      height: 3px;
      background: #eee;
      border-radius: 2px;
      margin-bottom: 8px;
    }}
    .sim-fill {{
      height: 100%;
      background: #4f6fff;
      border-radius: 2px;
    }}

    .result-content {{
      font-size: 12px;
      line-height: 1.6;
      color: #555;
      display: -webkit-box;
      -webkit-line-clamp: 4;
      -webkit-box-orient: vertical;
      overflow: hidden;
    }}

    .sim-score {{
      font-size: 10px;
      color: #bbb;
      margin-top: 6px;
    }}

    .no-matches {{
      font-size: 12px;
      color: #ccc;
      font-style: italic;
    }}
  </style>
</head>
<body>
  <div class="layout">
    <div class="articles-panel">
      <div class="panel-title">Article Chunks</div>
      <div id="articles"></div>
    </div>
    <div class="results-panel">
      <div class="panel-title">Top Matches</div>
      <div id="results">
        <p class="placeholder">Hover over a chunk on the left to see its top 3 matching patent descriptions and claims.</p>
      </div>
    </div>
  </div>

  <script>
    const DATA = {data_json};
    const ARTICLES = {articles_json};
    const PATENT_TITLES = {titles_json};
    const ARTICLE_URLS = {article_urls_json};

    const articlesEl = document.getElementById('articles');
    let activeChunk = null;

    for (const [articleName, chunks] of Object.entries(ARTICLES)) {{
      const section = document.createElement('div');
      section.className = 'article-section';

      const title = document.createElement('div');
      title.className = 'article-title';
      const url = ARTICLE_URLS[articleName];
      title.innerHTML = escapeHtml(articleName.replace(/_/g, ' ')) +
        (url ? ' <a class="article-source-link" href="' + url + '" target="_blank" rel="noopener">↗ source</a>' : '');
      section.appendChild(title);

      for (const [idx, chunkName] of chunks) {{
        const chunk = DATA[chunkName];
        const div = document.createElement('div');
        div.className = 'chunk';
        div.dataset.chunk = chunkName;
        div.innerHTML = '<div class="chunk-label">Chunk ' + idx + '</div>' + escapeHtml(chunk.chunk_content);
        div.addEventListener('mouseenter', () => showResults(chunkName, div));
        section.appendChild(div);
      }}

      articlesEl.appendChild(section);
    }}

    function showResults(chunkName, el) {{
      if (activeChunk) activeChunk.classList.remove('active');
      el.classList.add('active');
      activeChunk = el;

      const chunk = DATA[chunkName];
      let html = '';

      html += '<div class="section-label">Top 3 Descriptions</div>';
      if (chunk.top_descriptions.length === 0) {{
        html += '<p class="no-matches">No description matches</p>';
      }} else {{
        for (const d of chunk.top_descriptions) {{
          html += resultCard(d.marker, d.patent_id, d.content, d.similarity);
        }}
      }}

      html += '<div class="section-label">Top 3 Claims</div>';
      if (chunk.top_claims.length === 0) {{
        html += '<p class="no-matches">No claim matches</p>';
      }} else {{
        for (const c of chunk.top_claims) {{
          html += resultCard(c.marker, c.patent_id, c.content, c.similarity);
        }}
      }}

      document.getElementById('results').innerHTML = html;
    }}

    function resultCard(marker, patentId, content, similarity) {{
      const pct = Math.round(similarity * 100);
      const title = PATENT_TITLES[patentId] || '';
      const espacenetUrl = 'https://worldwide.espacenet.com/patent/search?q=pn%3D' + encodeURIComponent(patentId);
      return (
        '<div class="result-card">' +
          '<div class="result-meta">' +
            '<span class="result-marker">' + escapeHtml(String(marker)) + '</span>' +
            '<span class="result-patent">' + escapeHtml(patentId) + '</span>' +
          '</div>' +
          (title ? '<div class="result-title">' + escapeHtml(title) + '</div>' : '') +
          '<a class="result-espacenet" href="' + espacenetUrl + '" target="_blank" rel="noopener">View on Espacenet ↗</a>' +
          '<div class="sim-bar"><div class="sim-fill" style="width:' + pct + '%"></div></div>' +
          '<div class="result-content">' + escapeHtml(content) + '</div>' +
          '<div class="sim-score">Similarity: ' + similarity.toFixed(3) + '</div>' +
        '</div>'
      );
    }}

    function escapeHtml(str) {{
      return str.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
    }}
  </script>
</body>
</html>
"""


def load_article_urls(filepath):
    """Parse article variable names and their # https://... URL comments from a Python source file."""
    with open(filepath, "r") as f:
        src = f.read()
    matches = re.findall(r'(\w+)\s*=\s*"""[\s\S]*?"""\s*\n#\s*(https?://\S+)', src)
    return {name: url for name, url in matches}


def load_patent_titles(json_dir="json"):
    """Scan all JSON files in json_dir and build a patent_id -> title mapping."""
    titles = {}
    if not os.path.isdir(json_dir):
        return titles
    for fname in os.listdir(json_dir):
        if not fname.endswith(".json"):
            continue
        with open(os.path.join(json_dir, fname), "r") as f:
            try:
                patents = json.load(f)
            except json.JSONDecodeError:
                continue
        if not isinstance(patents, list):
            continue
        for p in patents:
            pid = p.get("patent_id")
            title = p.get("title")
            if pid and title:
                titles[pid] = title
    return titles


def generate(input_json, output_html, articles_py):
    with open(input_json, "r") as f:
        data = json.load(f)

    inverted = invert_data(data)
    articles = group_by_article(inverted)
    patent_titles = load_patent_titles()
    article_urls = load_article_urls(articles_py)

    html = HTML_TEMPLATE.format(
        data_json=json.dumps(inverted, ensure_ascii=False),
        articles_json=json.dumps(articles, ensure_ascii=False),
        titles_json=json.dumps(patent_titles, ensure_ascii=False),
        article_urls_json=json.dumps(article_urls, ensure_ascii=False),
    )

    with open(output_html, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"Written to {output_html}")
    webbrowser.open(f"file://{os.path.abspath(output_html)}")


def main():
    generate(
        "patent_article_similarities.json",
        "morphowave_visualisation.html",
        "context/idemia_articles.py",
    )
    generate(
        "ghostbat_patent_article_similarities.json",
        "ghostbat_visualisation.html",
        "context/ghostbat_articles.py",
    )


main()