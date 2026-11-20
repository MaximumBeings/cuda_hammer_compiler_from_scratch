#!/usr/bin/env python3
"""Check publication structure, code embeds, local links, diagrams and output scope."""
from pathlib import Path
import argparse, json, re, sys, urllib.parse, xml.etree.ElementTree as ET
from validation_manifest import CoverageError, require_inputs
ROOT=Path(__file__).resolve().parents[1];DOCS=ROOT/'docs'
def outside_code(text):return re.sub(r'^```[^\n]*\n.*?^```\s*$', '', text, flags=re.M|re.S)
def check(site=None):
 errors=[];outputs=0
 try:
  pages=require_inputs(ROOT,'pages'); sources=require_inputs(ROOT,'sources')
 except CoverageError as ex:
  print('Coverage failure: '+str(ex)); return 1
 for p in pages:
  s=p.read_text();plain=outside_code(s);headings=re.findall(r'^##\s+((?:\d+|[A-F])\.\d+)\b',plain,re.M)
  if len(headings)!=len(set(headings)):errors.append(f'{p.relative_to(ROOT)}: repeated numbered sections')
  for dest in re.findall(r'\]\(([^)]+)\)',plain):
   dest=urllib.parse.unquote(dest.split('#')[0].split('?')[0])
   if not dest or re.match(r'^[a-zA-Z]+:',dest) or dest.startswith('/'):continue
   if not (p.parent/dest).exists():errors.append(f'{p.relative_to(ROOT)}: missing link {dest}')
  for block in re.findall(r'^```text\n(.*?)^```',s,re.M|re.S):
   lines=[x.strip() for x in block.splitlines() if x.strip()]
   if lines and all(re.fullmatch(r'(?:\d+/\d+ checks passed|ALL CHECKS PASSED|ALL PASS)',x) for x in lines):errors.append(f'{p.relative_to(ROOT)}: pass-only sample output')
   outputs+=1
 # A source file should have a byte-equivalent fenced listing in its own part.
 for source in sources:
  if not source.is_file():continue
  content=source.read_text().strip();part=source.parent.parent
  blocks=[]
  for page in part.glob('*.md'):blocks.extend(re.findall(r'^```[^\n]*\n(.*?)^```',page.read_text(),re.M|re.S))
  if not any(content==b.strip() for b in blocks):errors.append(f'{source.relative_to(ROOT)}: source embed differs or absent')
 manifest=ROOT/'tools/diagram-manifest.json'
 if not manifest.is_file():
  print('Missing required diagram manifest'); return 1
 if manifest.exists():
  assets=json.loads(manifest.read_text())
  if not isinstance(assets,list) or not assets or len(assets)!=len(set(assets)):
   print('Expected nonempty unique diagram manifest'); return 1
  if len(assets)!=159:
   print('Diagram coverage differs from the 159-figure book contract'); return 1
  for asset in assets:
   p=DOCS/asset
   try:
    svg=ET.parse(p).getroot();assert svg.get('viewBox','').split()[2]=='760';assert svg.get('role')=='img'
    assert svg.find('{http://www.w3.org/2000/svg}title') is not None
    assert '#0b2e63' in p.read_text()
   except Exception as ex:errors.append(f'{asset}: invalid SVG {ex}')
   if site and not (Path(site)/asset).exists():errors.append(f'{asset}: missing built asset')
  print(f'Checked {len(assets)} accessible deep-blue SVGs.')
 print(f'Checked {len(pages)} pages, {len(sources)} source listings, {outputs} text blocks.')
 if errors:
  print('\n'.join(errors));return 1
 print('PASS: publication structure, code embeds, links and sample-output checks');return 0
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--site');args=parser.parse_args();sys.exit(check(args.site))
