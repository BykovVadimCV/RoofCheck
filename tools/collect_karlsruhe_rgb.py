#!/usr/bin/env python3
"""Fetch only RGB JPEGs from a public CC-BY Karlsruhe drone ZIP via HTTP ranges."""
import argparse,csv,hashlib,io,struct,time,zlib,zipfile
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
from urllib.request import Request,urlopen
from urllib.error import HTTPError
from PIL import Image,ImageOps,ImageDraw

URL='https://zenodo.org/records/7360996/files/Flug_105.zip?download=1'
SIZE=1246391459
UA='RoofCheck-Research/1.0 (open rooftop dataset; contact via Zenodo record)'
FIELDS=['id','file','source_entry','source_url','author','license','license_url','original_width','original_height','output_width','output_height','original_crc32','sha256','review']
def fetch(a,b):
 for n in range(4):
  try:
   req=Request(URL,headers={'User-Agent':UA,'Range':f'bytes={a}-{b}'})
   with urlopen(req,timeout=120) as response:
    if response.status!=206:raise ValueError('Server did not honor byte range')
    data=response.read()
   if len(data)!=b-a+1:raise ValueError('Short response')
   return data
  except HTTPError as e:
   if e.code==429:time.sleep(min(60,int(e.headers.get('Retry-After','60'))))
   elif n==3:raise
   else:time.sleep(2**n)
  except Exception:
   if n==3:raise
   time.sleep(2**n)
 raise RuntimeError('Range download failed')
class Remote(io.RawIOBase):
 def __init__(self):self.pos=0
 def seekable(self):return True
 def readable(self):return True
 def tell(self):return self.pos
 def seek(self,off,whence=0):self.pos=off if whence==0 else self.pos+off if whence==1 else SIZE+off;return self.pos
 def read(self,n=-1):
  if n<0:n=SIZE-self.pos
  if n<=0:return b''
  data=fetch(self.pos,min(SIZE-1,self.pos+n-1));self.pos+=len(data);return data

def save_rows(path,rows):
 tmp=path.with_suffix('.tmp')
 with tmp.open('w',encoding='utf-8',newline='') as f:w=csv.DictWriter(f,fieldnames=FIELDS);w.writeheader();w.writerows(rows.values())
 tmp.replace(path)

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--limit',type=int,default=352);p.add_argument('--workers',type=int,default=4);a=p.parse_args()
 if not 1<=a.workers<=6 or not 1<=a.limit<=352:p.error('workers 1..6; limit 1..352')
 root=a.output;root.mkdir(parents=True,exist_ok=True);images=root/'images';images.mkdir(exist_ok=True);manifest=root/'manifest.csv';rows={}
 if manifest.exists():
  with manifest.open(encoding='utf-8',newline='') as f:rows={r['source_entry']:r for r in csv.DictReader(f)}
 z=zipfile.ZipFile(Remote());entries=sorted(z.infolist(),key=lambda x:x.header_offset);end_by_offset={info.header_offset:(entries[i+1].header_offset if i+1<len(entries) else z.start_dir) for i,info in enumerate(entries)}
 rgb=[i for i in entries if '/RGB/' in i.filename and i.filename.lower().endswith('.jpg')][:a.limit]
 # Adjacent RGB files are fetched together; each block is one range request.
 blocks=[]
 for i in range(0,len(rgb),8):
  subset=rgb[i:i+8]
  if all(x.filename in rows and (root/rows[x.filename]['file']).is_file() for x in subset):continue
  blocks.append(subset)
 print('RGB entries:',len(rgb),'already saved:',len(rows),'remaining blocks:',len(blocks),flush=True)
 def get_block(block):
  start=block[0].header_offset;end=end_by_offset[block[-1].header_offset]-1;data=fetch(start,end);results=[]
  for entry in block:
   if entry.filename in rows and (root/rows[entry.filename]['file']).is_file():continue
   offset=entry.header_offset-start
   header=struct.unpack_from('<4s5H3I2H',data,offset)
   if header[0]!=b'PK\x03\x04':raise ValueError('Invalid ZIP local header')
   begin=offset+30+header[-2]+header[-1];comp=data[begin:begin+entry.compress_size]
   raw=zlib.decompress(comp,-15) if entry.compress_type==zipfile.ZIP_DEFLATED else comp if entry.compress_type==zipfile.ZIP_STORED else None
   if raw is None or len(raw)!=entry.file_size or (zlib.crc32(raw)&0xffffffff)!=entry.CRC:raise ValueError('ZIP CRC/size failed: '+entry.filename)
   name=Path(entry.filename).stem;dst=images/(name+'.jpg')
   with Image.open(io.BytesIO(raw)) as im:
    original=im.size;im=ImageOps.exif_transpose(im).convert('RGB');im.thumbnail((1600,1600),Image.Resampling.LANCZOS);output=im.size;im.save(dst,quality=94)
   results.append({'id':name,'file':'images/'+dst.name,'source_entry':entry.filename,'source_url':'https://zenodo.org/records/7360996','author':'Kahn, Mayer, Hou, Beiersdörfer, Götz, Volk','license':'CC BY 4.0','license_url':'https://creativecommons.org/licenses/by/4.0/','original_width':original[0],'original_height':original[1],'output_width':output[0],'output_height':output[1],'original_crc32':f'{entry.CRC:08x}','sha256':hashlib.sha256(dst.read_bytes()).hexdigest(),'review':'UNREVIEWED: snow and viewpoint require visual review; same buildings recur'})
  return results
 with ThreadPoolExecutor(max_workers=a.workers) as pool:
  for future in as_completed([pool.submit(get_block,b) for b in blocks]):
   try:
    for row in future.result():rows[row['source_entry']]=row
    save_rows(manifest,rows);print('Saved:',len(rows),'/',len(rgb),flush=True)
   except Exception as e:print('BLOCK FAILED:',str(e),flush=True)
 items=list(rows.values());preview=root/'contact_sheets';preview.mkdir(exist_ok=True)
 for start in range(0,len(items),48):
  sheet=Image.new('RGB',(1500,1480),'white');d=ImageDraw.Draw(sheet)
  for i,row in enumerate(items[start:start+48]):
   with Image.open(root/row['file']) as im:im=ImageOps.contain(im.convert('RGB'),(245,155))
   x=i%6*250;y=i//6*185;sheet.paste(im,(x,y+24));d.text((x+3,y+3),row['id'],fill='black')
  sheet.save(preview/f'{start//48+1:03d}.jpg',quality=87)
 print('DONE',len(rows),'images',flush=True)
if __name__=='__main__':main()
