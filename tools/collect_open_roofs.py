#!/usr/bin/env python3
"""Collect openly licensed Commons roof image candidates, not a verified snow-free dataset."""
import argparse, csv, hashlib, html, json, re, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from urllib.error import HTTPError
from PIL import Image, ImageDraw, ImageOps

API='https://commons.wikimedia.org/w/api.php'
UA='RoofCheck-OpenRoofResearch/1.0 (public photo collection; Wikimedia Commons API)'
QUERIES=['standing seam metal roof','aerial metal roof building','drone metal roof house','aerial roof building',
         'rooftops aerial view city','view over rooftops city','rooftop view buildings','metal roof rooftop',
         'flat roof building rooftop','roof construction rooftop','standing seam roof aerial',
         'roof tiles rooftops aerial','rooftops from building','industrial roof aerial','крыши вид сверху','крыша кровля металл']
BANNED=re.compile(r'\b(snow|snowy|snowfall|winter|wintry|snow-covered|снег\w*|зим\w*|neige|schnee)\b',re.I)
TAGS=re.compile('<[^>]+>')
FIELDS=['id','title','file','page_url','original_url','download_url','author','license','license_url','attribution','source_query','width','height','sha256','screening']
def get(url):
 for retry in range(4):
  try:
   with urlopen(Request(url,headers={'User-Agent':UA}),timeout=40) as r:return r.read()
  except HTTPError as exc:
   if exc.code==429:raise
   if retry==3:raise
   time.sleep(1.5*(retry+1))
  except Exception:
   if retry==3:raise
   time.sleep(1.5*(retry+1))
def plain(x):return html.unescape(TAGS.sub('',str(x or ''))).strip()
def meta(d,k):return plain(d.get(k,{}).get('value',''))
def collect(out,limit):
 out.mkdir(parents=True,exist_ok=True); imgs=out/'images';imgs.mkdir(exist_ok=True)
 index=out/'manifest.csv'; old={}
 if index.exists():
  with index.open(encoding='utf-8-sig',newline='') as f:old={r['id']:r for r in csv.DictReader(f)}
 candidates={}
 for q in QUERIES:
  if len(candidates)>=limit*3:break
  par={'action':'query','generator':'search','gsrsearch':q+' -snow -winter','gsrnamespace':6,'gsrlimit':50,'prop':'imageinfo','iiprop':'url|extmetadata|size','iiurlwidth':1200,'format':'json'}
  try:data=json.loads(get(API+'?'+urlencode(par)))
  except Exception as e:
   print('SEARCH FAILED',q,str(e),flush=True)
   if isinstance(e,HTTPError) and e.code==429:break
   continue
  for p in data.get('query',{}).get('pages',{}).values():
   info=(p.get('imageinfo') or [{}])[0];m=info.get('extmetadata',{});lic=meta(m,'LicenseShortName');title=p.get('title','');desc=meta(m,'ImageDescription');
   if not re.search(r'(roof|rooftop|кры[шв]|кровл)',title+' '+desc,re.I):continue
   if BANNED.search(title+' '+desc):continue
   if not (lic=='CC0' or re.fullmatch(r'CC BY(?:-SA)? [234](?:\.\d)?',lic) or lic in ('Public domain','Public Domain')):continue
   if not info.get('thumburl') or not info.get('url'):continue
   if info.get('width',0)<600 or info.get('height',0)<300:continue
   id=str(p['pageid']); candidates.setdefault(id,dict(id=id,title=title,page_url=info.get('descriptionurl',''),original_url=info['url'],download_url=info['thumburl'],author=meta(m,'Artist'),license=lic,license_url=meta(m,'LicenseUrl'),attribution=meta(m,'Attribution'),source_query=q,width=info['width'],height=info['height']))
  print('Search:',q,'| candidates:',len(candidates),flush=True)
  time.sleep(.2)
 pending=[x for x in candidates.values() if x['id'] not in old][:max(0,limit-len(old))]
 def download(row):
  data=get(row['download_url']);fn=row['id']+'.jpg';path=imgs/fn
  from io import BytesIO
  with Image.open(BytesIO(data)) as im:
   im=ImageOps.exif_transpose(im).convert('RGB');im.save(path,quality=95)
  row.update(file='images/'+fn,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),screening='UNREVIEWED: license/source metadata checked; roof and no-snow need visual review')
  return row
 count=0
 with ThreadPoolExecutor(max_workers=3) as pool:
  futures=[pool.submit(download,row) for row in pending]
  for fut in as_completed(futures):
   try:
    row=fut.result(); old[row['id']]=row;count+=1
    if count%25==0:print('Downloaded:',count,'| total:',len(old),flush=True)
   except Exception as e:
    print('DOWNLOAD FAILED:',str(e)[:150],flush=True)
    if isinstance(e,HTTPError) and e.code==429:
     for future in futures:future.cancel()
     print('Stopped new requests due to Wikimedia rate limit',flush=True)
     break
   with index.open('w',encoding='utf-8',newline='') as f:w=csv.DictWriter(f,fieldnames=FIELDS);w.writeheader();w.writerows(old.values())
 with index.open('w',encoding='utf-8',newline='') as f:w=csv.DictWriter(f,fieldnames=FIELDS);w.writeheader();w.writerows(old.values())
 sheets=out/'contact_sheets';sheets.mkdir(exist_ok=True);items=list(old.values())
 for start in range(0,len(items),60):
  page=Image.new('RGB',(1500,1800),'white');draw=ImageDraw.Draw(page)
  for i,row in enumerate(items[start:start+60]):
   p=out/row['file']
   if not p.exists():continue
   with Image.open(p) as im:thumb=ImageOps.contain(im.convert('RGB'),(245,155))
   x=(i%6)*250;y=(i//6)*180;page.paste(thumb,(x,y+20));draw.text((x+4,y+2),row['id'],fill='black')
  page.save(sheets/f'{start//60+1:03d}.jpg',quality=85)
 print('DONE',len(old),'manifest',index,flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--limit',type=int,default=500);a=p.parse_args();collect(a.output,a.limit)
