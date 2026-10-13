from PIL import Image, ImageDraw, ImageFont
import xml.etree.ElementTree as ET
import re,html
from pathlib import Path
root=ET.parse(Path(__file__).resolve().parent / 'H200Cluster-full-topology.drawio')
cells=root.findall('.//mxCell'); byid={c.get('id'):c for c in cells}; scale=2
model=root.find('.//mxGraphModel')
im=Image.new('RGB',(int(model.get('pageWidth'))*scale,int(model.get('pageHeight'))*scale),'white'); draw=ImageDraw.Draw(im)
font='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
def styles(c): return dict(p.split('=',1) for p in c.get('style','').split(';') if '=' in p)
def geom(c):
 g=c.find('mxGeometry');return [float(g.get(k,0)) for k in ['x','y','width','height']]
def xy(v): return tuple(round(n*scale) for n in v)
def label(value,x,y,w,size=14,center=False):
 text=html.unescape(re.sub('<[^>]+>','',value.replace('<br>','\n')))
 f=ImageFont.truetype(font,round(size*scale)); lines=[]
 for paragraph in text.split('\n'):
  line=''
  for ch in paragraph:
   if draw.textlength(line+ch,font=f)>(w-16)*scale and line: lines.append(line);line=ch
   else:line+=ch
  lines.append(line)
 for line in lines:
  xx=x*scale+(w*scale-draw.textlength(line,font=f))/2 if center else (x+8)*scale
  draw.text((xx,y*scale),line,font=f,fill='#172033');y+=size*1.5
# Background
c=byid['node'];x,y,w,h=geom(c);draw.rounded_rectangle(xy((x,y,x+w,y+h)),radius=20,fill='#F8FAFC',outline='#CBD5E1',width=2)
# edges behind boxes
for c in cells:
 if c.get('edge')!='1':continue
 s=styles(c);a=geom(byid[c.get('source')]);b=geom(byid[c.get('target')])
 start=(a[0]+a[2]*float(s.get('exitX','.5')),a[1]+a[3]*float(s.get('exitY','.5')))
 end=(b[0]+b[2]*float(s.get('entryX','.5')),b[1]+b[3]*float(s.get('entryY','.5')))
 points=[(float(p.get('x')),float(p.get('y'))) for p in c.findall('.//mxPoint')]
 path=[start]+points+[end]
 draw.line([xy(p) for p in path],fill=s.get('strokeColor','#64748B'),width=3)
 # Short labels are centered on their longest vertical/horizontal segment.
 if not c.get('value'):continue
 segments=list(zip(path,path[1:])); p,q=max(segments,key=lambda z:abs(z[1][0]-z[0][0])+abs(z[1][1]-z[0][1]))
 lx=(p[0]+q[0])/2;ly=(p[1]+q[1])/2
 txt=html.unescape(re.sub('<[^>]+>','',c.get('value','')))
 f=ImageFont.truetype(font,22); tw=draw.textlength(txt,font=f)
 xx=lx*scale-tw/2; yy=ly*scale-15
 draw.rectangle((xx-4,yy,xx+tw+4,yy+30),fill='white');draw.text((xx,yy),txt,font=f,fill=s.get('strokeColor','#64748B'))
for c in cells:
 if c.get('vertex')!='1' or c.get('id')=='node':continue
 x,y,w,h=geom(c);s=styles(c);size=float(s.get('fontSize','14'))
 if s.get('fillColor')!='none':
  draw.rounded_rectangle(xy((x,y,x+w,y+h)),radius=14,fill=s.get('fillColor','white'),outline=s.get('strokeColor','#CBD5E1'),width=2)
  label(c.get('value',''),x,y+8,w,size,True)
 else:label(c.get('value',''),x,y,w,size)
im.save(Path(__file__).resolve().parent / 'H200Cluster-full-topology.png')
print(f'PNG exported: {im.width} × {im.height}')
