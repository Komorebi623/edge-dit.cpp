#!/usr/bin/env python3
"""CPU artifact QA: actual PPTX package and rendered pages, no fabricated status."""
import argparse,json,re,hashlib,zipfile
from pathlib import Path
import xml.etree.ElementTree as E
from PIL import Image,ImageOps,ImageDraw
NS={'p':'http://schemas.openxmlformats.org/presentationml/2006/main','a':'http://schemas.openxmlformats.org/drawingml/2006/main'}
def check(directory):
 r=Path(directory);ppt=r/'H3-CUDA-优化汇报.pptx';pdf=r/'preview/H3-CUDA-优化汇报.pdf';data=json.loads((r/'report-data.json').read_text());errors=[]
 with zipfile.ZipFile(ppt) as z:
  names=sorted([n for n in z.namelist() if re.fullmatch(r'ppt/slides/slide\d+\.xml',n)],key=lambda n:int(re.search(r'(\d+)\.xml',n)[1]));texts=[]
  dims=E.fromstring(z.read('ppt/presentation.xml')).find('p:sldSz',NS);cx,cy=int(dims.attrib['cx']),int(dims.attrib['cy'])
  for name in names:
   tree=E.fromstring(z.read(name));text='\n'.join(n.text or '' for n in tree.findall('.//a:t',NS));texts.append(text)
   if re.search(r'\b(lorem|ipsum|TODO|undefined|NaN)\b',text,re.I):errors.append(name+': placeholder/invalid value')
   for shape in tree.findall('.//p:sp',NS)+tree.findall('.//p:pic',NS):
    tr=shape.find('.//a:xfrm',NS)
    if tr is None:continue
    off=tr.find('a:off',NS);ext=tr.find('a:ext',NS)
    if off is None or ext is None:continue
    x,y=int(off.attrib['x']),int(off.attrib['y']);w,h=int(ext.attrib['cx']),int(ext.attrib['cy'])
    if x<0 or y<0 or x+w>cx+100 or y+h>cy+100:errors.append(name+': shape outside canvas')
  alltext='\n\n'.join(texts)
  for key in ['Cache','SubBlock','LoRA','AdaLN']:
   if key.lower() not in alltext.lower():errors.append('missing topic '+key)
  (r/'extracted-slides.txt').write_text(alltext)
 images=sorted((r/'preview').glob('slide-*.jpg'))
 if len(images)!=len(names):errors.append(f'PDF preview pages {len(images)} != PPT slides {len(names)}')
 for image in images:
  with Image.open(image) as im:
   if min(im.size)<500:errors.append('render unexpectedly small '+image.name)
 if not pdf.is_file() or pdf.stat().st_size<1000:errors.append('rendered PDF absent')
 # Read-only QA contact sheets, no overwritten previews.
 for page,start in enumerate(range(0,len(images),6),1):
  items=images[start:start+6];sheet=Image.new('RGB',(1000,3*315),'#e6e9eb');draw=ImageDraw.Draw(sheet)
  for i,path in enumerate(items):
   with Image.open(path) as im:thumb=ImageOps.contain(im,(490,280));x=(i%2)*500+5;y=(i//2)*315+25;sheet.paste(thumb,(x,y));draw.text((x,y-20),f'Slide {start+i+1}',fill='black')
  sheet.save(r/'preview'/f'contact-{page:02d}.jpg')
 result={'status':'passed' if not errors else 'failed','slide_count':len(names),'rendered_pages':len(images),'errors':errors,'pptx_sha256':hashlib.sha256(ppt.read_bytes()).hexdigest(),'pdf_sha256':hashlib.sha256(pdf.read_bytes()).hexdigest(),'as_of':data['as_of'],'visual_review':'Automated package/bounds/render checks; initial draft manually reviewed separately. Updated overnight result values still require final human glance.'}
 (r/'qa.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n');print(json.dumps(result,ensure_ascii=False))
 if errors:raise SystemExit(1)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--dir',type=Path,required=True);a=p.parse_args();check(a.dir.resolve())
