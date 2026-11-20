#!/usr/bin/env python3
"""Build accessible 760-unit SVGs and idempotent embeds, excluding code fences."""
from pathlib import Path
import hashlib, html, json, re, textwrap
from diagram_concepts import ROWS, APPENDICES
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'docs/images'
BG='#0b2e63'; PANEL='#16457e'; BORDER='#5b83b5'; INK='#f5f4fa'; MUTED='#bdc9df'; ACCENT='#8ed5c4'
def esc(x):return html.escape(str(x),quote=True)
def text(x,y,s,size=20,color=INK,weight=400):
 return f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" font-weight="{weight}">{esc(s)}</text>'
def wrap(x,y,s,size=20,width=52,color=INK,weight=400):
 lines=textwrap.wrap(s,width,break_long_words=False,break_on_hyphens=False)
 return ''.join(text(x,y+i*size*1.4,line,size,color,weight) for i,line in enumerate(lines)),len(lines)*size*1.4

def box(x,y,w,h,fill=PANEL):return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="12" fill="{fill}" stroke="{BORDER}" stroke-width="1.5"/>'
def arrow(y):return f'<path d="M380 {y} v30" fill="none" stroke="{ACCENT}" stroke-width="3" marker-end="url(#arrow)"/>'
def save(name,title,body,height,desc):
 path=OUT/(name+'.svg');path.parent.mkdir(parents=True,exist_ok=True)
 svg=f'''<svg xmlns="http://www.w3.org/2000/svg" width="760" height="{height}" viewBox="0 0 760 {height}" role="img" aria-labelledby="title desc"><title id="title">{esc(title)}</title><desc id="desc">{esc(desc)}</desc><defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0 0 L10 5 L0 10Z" fill="{ACCENT}"/></marker></defs><g font-family="Arial, Helvetica, sans-serif">{box(1,1,758,height-2,BG)}{body}</g></svg>'''
 path.write_text(svg);return path,hashlib.sha256(svg.encode()).hexdigest()[:12]
def diamond(name,section):
 title='One shared value, two consumers'
 b=text(32,38,f'CUDA HAMMER / CONCEPT {section}',14,ACCENT,700)
 b+=text(32,88,title,29,INK,600)
 nodes=[(70,145,'input a'),(440,145,'input b'),(255,275,'t1 = a + b'),(70,420,'t2 = t1 * c'),(440,420,'t3 = ReLU(t1)'),(255,565,'out = t2 + t3')]
 for x,y,label in nodes:b+=box(x,y,250,72)+text(x+18,y+44,label,23,INK,500)
 b+=box(70,275,150,72)+text(88,319,"input c",23,INK,500)
 b+=' <path d="M145 350 V417" fill="none" stroke="'+ACCENT+'" stroke-width="3" marker-end="url(#arrow)"/>'
 for x1,y1,x2,y2 in [(195,220,350,272),(565,220,410,272),(320,350,195,417),(440,350,565,417),(195,495,350,562),(565,495,410,562)]:
  b+=f'<path d="M{x1} {y1} L{x2} {y2}" fill="none" stroke="{ACCENT}" stroke-width="3" marker-end="url(#arrow)"/>'
 note='t1 has two users. Preserve reuse or account for recomputation when forming fusion groups.'
 chunk,h=wrap(52,695,note,18,66,MUTED);b+=box(32,660,696,h+48,'#123e68')+chunk
 return save(name,title,b,int(730+h),note)

def concept(name,section,spec):
 if section in {'4.1','13.2','25.1'}:return diamond(name,section)
 title,stages,note=spec;b=text(32,38,f'CUDA HAMMER / CONCEPT {section}',14,ACCENT,700)
 chunk,h=wrap(32,84,title,27,43,weight=600);b+=chunk;y=100+h
 for i,label in enumerate(stages.split('|')):
  chunk,h=wrap(72,y+39,label,23,43,weight=500);bh=max(84,h+44)
  b+=box(40,y,680,bh)+text(55,y+27,str(i+1),13,ACCENT,700)+chunk;y+=bh
  if i<2:b+=arrow(y+4);y+=42
 chunk,h=wrap(52,y+61,note,18,66,MUTED)
 b+=box(32,y+26,696,h+44,'#123e68')+chunk
 return save(name,title,b,int(y+h+96),stages.replace('|','; ')+'. '+note)
def opener(name,title,section,topics):
 b=text(32,38,f'CUDA HAMMER / {section}',14,ACCENT,700)
 chunk,h=wrap(32,91,title,30,39,weight=600);b+=chunk;y=110+h
 for i,topic in enumerate(topics):
  chunk,h=wrap(76,y+34,topic,22,47);bh=max(82,h+35)
  b+=box(32,y,696,bh)+text(45,y+28,str(i+1),14,ACCENT,700)+chunk;y+=bh+18
 b+=text(32,y+19,'Read the contracts. Run the code. Check the result.',17,MUTED)
 return save(name,title,b,int(y+48),'Chapter roadmap: '+ '; '.join(topics))
def headings(s):
 result=[];fence=False
 for line in s.splitlines():
  if line.startswith('```'):fence=not fence;continue
  if not fence and re.match(r'^#{1,2} ',line):result.append(line)
 return result
def embed(page,key,asset,alt,caption):
 path,digest=asset;rel=Path(__import__('os').path.relpath(path,page.parent)).as_posix()
 return f'\n<!-- hammer:figure:{key}:start -->\n<figure class="chapter-figure" markdown="1">\n\n[![{alt}]({rel}?v={digest})]({rel}?v={digest})\n\n<figcaption>{caption}</figcaption>\n</figure>\n<!-- hammer:figure:{key}:end -->\n'
def inject(s,blocks):
 s=re.sub(r'\n<!-- hammer:figure:.*?:start -->.*?<!-- hammer:figure:.*?:end -->\n','',s,flags=re.S)
 result=[];fence=False
 for line in s.splitlines(keepends=True):
  result.append(line)
  if line.startswith('```'):fence=not fence
  if not fence and line.rstrip() in blocks:result.append(blocks[line.rstrip()])
 return ''.join(result)
def build():
 records=[]
 for page in sorted((ROOT/'docs').glob('part*/*.md')):
  s=page.read_text();hs=headings(s);m=re.search(r'(\d+)',page.name);n=int(m[1]);h1=next(h for h in hs if h.startswith('# '));title=re.sub(r'^(?:Chapter )?\d+[.:]?\s*','',h1[2:])
  sections=[h for h in hs if re.match(r'^## '+str(n)+r'\.\d+\b',h)]
  sections=list(dict.fromkeys(sections));assert len(sections)==3,(page,sections)
  topics=[re.sub(r'^## \d+\.\d+\s*(?:--)?\s*','',x) for x in sections]
  asset=opener(f'openers/{n:02}',title,f'CHAPTER {n:02}',topics)
  blocks={h1:embed(page,f'opener-{n}',asset,title,'Chapter roadmap. Open any diagram for a full-size view.')};records.append(str(asset[0].relative_to(ROOT/'docs')))
  for index,h in enumerate(sections):
   spec=ROWS[n][index];asset=concept(f'concepts/{n:02}-{index+1:02}',f'{n}.{index+1}',spec)
   blocks[h]=embed(page,f'{n}.{index+1}',asset,spec[0],spec[2]);records.append(str(asset[0].relative_to(ROOT/'docs')))
  page.write_text(inject(s,blocks))
 for page in sorted((ROOT/'docs/appendix').glob('*.md')):
  letter=page.name[0].upper();s=page.read_text();hs=headings(s);h1=next(h for h in hs if h.startswith('# '));title=re.sub(r'^(?:Appendix )?[A-F][.:]?\s*(?:--)?\s*','',h1[2:])
  sections=[h for h in hs if re.match(r'^## '+letter+r'\.\d+\b',h)]
  topics=[re.sub(r'^## [A-F]\.\d+\s*(?:--)?\s*','',h) for h in sections] or ['Review assumptions','Trace concrete examples','Compare with worked solutions']
  asset=opener('openers/appendix-'+letter.lower(),title,'APPENDIX '+letter,topics)
  blocks={h1:embed(page,'appendix-'+letter,asset,title,'Appendix roadmap.')} ;records.append(str(asset[0].relative_to(ROOT/'docs')))
  for i,h in enumerate(sections):
   base=APPENDICES[letter];spec=(topics[i],base[1],base[2]);asset=concept(f'concepts/{letter.lower()}-{i+1:02}',f'{letter}.{i+1}',spec)
   blocks[h]=embed(page,f'{letter}.{i+1}',asset,spec[0],spec[2]);records.append(str(asset[0].relative_to(ROOT/'docs')))
  page.write_text(inject(s,blocks))
 title='From a tensor graph to a validated implementation';b=text(32,38,'CUDA HAMMER / THE WHOLE BOOK',14,ACCENT,700)
 chunk,h=wrap(32,85,title,29,41,weight=600);b+=chunk;y=106+h
 steps=[('01 / Understand the cost','Materialized arrays, launches and a qualified performance model.'),('02 / Build and validate the IR','Parse names, infer shapes, inspect dependencies and print the graph.'),('03 / Transform the program','Fold constants, remove dead work, reuse expressions and form legal fusion groups.'),('04 / Generate and execute code','Scalar loops, target-specific SIMD, CUDA kernels and a checked JIT boundary.'),('05 / Tune with context','Search schedules, measure correct candidates and cache under sufficient target metadata.'),('06 / Compare and apply','Production compiler case studies, fixed pipelines, independent references and explicit limits.')]
 for i,(label,detail) in enumerate(steps):
  a,ah=wrap(54,y+36,label,23,49,weight=600);c,ch=wrap(54,y+73,detail,18,63,MUTED);bh=ah+ch+62;b+=box(32,y,696,bh)+a+c;y+=bh
  if i<len(steps)-1:b+=arrow(y+4);y+=42
 b+=text(32,y+39,'Compile checks != host execution != GPU execution.',20,ACCENT,600)
 asset=save('book-overview',title,b,int(y+73),'A six-stage book map with validation boundaries.');page=ROOT/'docs/index.md';s=page.read_text();h1=headings(s)[0];page.write_text(inject(s,{h1:embed(page,'book-overview',asset,title,'Compiler structure and validation form one continuous workflow.')}));records.append(str(asset[0].relative_to(ROOT/'docs')))
 (ROOT/'tools/diagram-manifest.json').write_text(json.dumps(records,indent=2)+'\n')
 print(f'Built {len(records)} full-width accessible deep-blue diagrams.')
if __name__=='__main__':build()
