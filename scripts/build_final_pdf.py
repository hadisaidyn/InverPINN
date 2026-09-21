"""Typeset the preserved manuscript and poster, without scientific computation.

Reads byte-verified figures/tables and raw results. Equations become numbered
native PDF text with unchanged signs/terms. No model, solver or checkpoint is
imported. Existing outputs are never overwritten; choose a new output root.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import importlib.metadata
import json
from pathlib import Path
import re
from statistics import median
import subprocess

from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen.canvas import Canvas
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Image,
                               Table, TableStyle, PageBreak, KeepTogether)

ROOT = Path(__file__).resolve().parents[1]
INK, BLUE, RED = '#192D3D', '#16688A', '#A63F26'
TITLE = 'InverPINN: Diagnosing Source-Strength Bias in Physics-Informed Neural Networks for Sparse Pollution Source Inversion'
AUTHOR = 'Khadis Aidyn'
EQUATIONS = [
 'C<sub>t</sub> + uC<sub>x</sub> + vC<sub>y</sub> = D(C<sub>xx</sub> + C<sub>yy</sub>) + S(x,y)<br/>S = QG = Q exp[-((x-x<sub>s</sub>)<super>2</super> + (y-y<sub>s</sub>)<super>2</super>)/(2σ<super>2</super>)]',
 'Δt (|u|/Δx + |v|/Δy + 2D/Δx<super>2</super> + 2D/Δy<super>2</super>) ≤ 1',
 'C<sub>θ</sub>(x,y,t) = 16x(1-x)y(1-y)(t/0.8) softplus(f<sub>θ</sub>(x,y,t))',
 'L = (10/C<sub>*</sub><super>2</super>) MSE(C<sub>θ</sub>-z) + (T<sub>*</sub><super>2</super>/C<sub>*</sub><super>2</super>) MSE(R<sub>0</sub>-QG)<br/>     + (1/C<sub>*</sub><super>2</super>) MSE(C<sub>θ</sub>|<sub>t=0</sub>) + (1/C<sub>*</sub><super>2</super>) MSE(C<sub>θ</sub>|<sub>∂Ω</sub>)',
 '∂<sub>Q</sub>L<sub>PDE</sub> = -2(T<sub>*</sub><super>2</super>/C<sub>*</sub><super>2</super>) ⟨G, R<sub>0</sub>-QG⟩',
 'Q<sub>unconstrained</sub><super>*</super> = (Σ<sub>i</sub> w<sub>i</sub>G<sub>i</sub>R<sub>0,i</sub>) / (Σ<sub>i</sub> w<sub>i</sub>G<sub>i</sub><super>2</super>)',
 'Q<sub>obs</sub><super>*</super> = max(0, h<super>T</super>z / (h<super>T</super>h))',
]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verified_inputs():
    """Fail on changed evidence; derive display medians from sealed CSVs."""
    evidence = ROOT/'paper/evidence'
    manifest = json.loads((evidence/'manifest.json').read_text())
    for name, record in manifest['files'].items():
        if sha(evidence/name) != record['sha256']:
            raise ValueError(f'Changed evidence: {name}')
    for name, digest in json.loads((ROOT/'paper/generated/artifact_hashes.json').read_text()).items():
        if sha(ROOT/'paper/generated'/name) != digest:
            raise ValueError(f'Changed canonical report: {name}')
    result = {}
    for method in ('J1', 'classical'):
        with (evidence/'final'/f'{method}_raw.csv').open() as f:
            rows = list(csv.DictReader(f))
        if len(rows) != 30: raise ValueError('All 30 scenarios must remain')
        result[method] = dict(recovered=sum(r['recovery_success'].lower()=='true' for r in rows),
            localization=f"{median(float(r['localization_error']) for r in rows):.6f}",
            Q=f"{100*median(float(r['relative_strength_error']) for r in rows):.3f}%",
            L2=f"{100*median(float(r['relative_l2']) for r in rows):.3f}%")
    if result['J1']['recovered']!=23 or result['classical']['recovered']!=30:
        raise ValueError('Frozen decision changed')
    return result


def register_fonts(font_dir=None):
    """Use DejaVu from an explicit directory or the pinned reporting install."""
    if font_dir is None:
        try:
            import matplotlib
            font_dir = Path(matplotlib.get_data_path())/'fonts/ttf'
        except ImportError as exc:
            raise RuntimeError('Install report requirements or pass --font-dir with DejaVu fonts') from exc
    folder = Path(font_dir)
    for label, prefix in [('Body','DejaVuSerif'),('Sans','DejaVuSans')]:
        for suffix, filename in [('',f'{prefix}.ttf'),('-Bold',f'{prefix}-Bold.ttf')]:
            pdfmetrics.registerFont(TTFont(label+suffix,str(folder/filename)))
        pdfmetrics.registerFontFamily(label,normal=label,bold=label+'-Bold',italic=label,boldItalic=label+'-Bold')
    return {p.name:sha(p) for prefix in ('DejaVuSerif','DejaVuSans') for p in
            (folder/f'{prefix}.ttf',folder/f'{prefix}-Bold.ttf')}


def rich(text):
    """Convert presentational Markdown, keeping citations and scientific text."""
    text=text.replace('−','-').replace('–','-').replace('—','-').replace('‑','-')
    text=re.sub(r'\[([^]]+)\]\(([^)]+)\)',r'\1',text)
    text=html.escape(text)
    text=re.sub(r'\*\*(.+?)\*\*',r'<b>\1</b>',text)
    # Asterisks attached to variables (C*, T*, Q*) are mathematical symbols,
    # not Markdown emphasis delimiters. Never consume them across prose.
    text=re.sub(r'(?<![\w*])\*([^*]+)\*(?![\w*])',r'<i>\1</i>',text)
    return re.sub(r'`([^`]+)`',r'\1',text)


def styles():
    return {
      'body':ParagraphStyle('body',fontName='Body',fontSize=10.1,leading=13.2,spaceAfter=4,alignment=TA_JUSTIFY,allowWidows=0,allowOrphans=0,textColor=colors.HexColor(INK)),
      'h1':ParagraphStyle('h1',fontName='Sans-Bold',fontSize=14,leading=18,textColor=colors.HexColor(BLUE),spaceBefore=13,spaceAfter=8,keepWithNext=True),
      'h2':ParagraphStyle('h2',fontName='Sans-Bold',fontSize=11,leading=15,textColor=colors.HexColor(INK),spaceBefore=10,spaceAfter=6,keepWithNext=True),
      'caption':ParagraphStyle('caption',fontName='Body',fontSize=9,leading=12,spaceAfter=12,textColor=colors.HexColor(INK)),
      'table':ParagraphStyle('table',fontName='Sans',fontSize=8.4,leading=11,textColor=colors.HexColor(INK)),
      'equation':ParagraphStyle('equation',fontName='Body',fontSize=10,leading=17,spaceAfter=6),
    }


def table_from_md(text,widths,style):
    rows=[line.strip().strip('|').split('|') for line in text.splitlines() if line.startswith('|')]
    rows=[r for r in rows if not all(re.fullmatch(r'[\s:-]+',c) for c in r)]
    t=Table([[Paragraph(rich(c.strip()),style) for c in row] for row in rows],colWidths=widths,repeatRows=1,hAlign='LEFT')
    t.setStyle(TableStyle([('VALIGN',(0,0),(-1,-1),'TOP'),('BACKGROUND',(0,0),(-1,0),colors.HexColor('#E8F0F3')),
       ('LINEBELOW',(0,0),(-1,0),0.8,colors.HexColor(BLUE)),('LINEBELOW',(0,-1),(-1,-1),0.5,colors.HexColor(BLUE)),
       ('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),7)]))
    return t


def figure(number,caption,width,sty):
    path=sorted((ROOT/'paper/generated/figures').glob(f'{number:02d}_*.png'))[0]
    with PILImage.open(path) as im: height=width*im.height/im.width
    return KeepTogether([Image(str(path),width=width,height=height),Spacer(1,5),Paragraph(rich(caption),sty['caption'])])


def bibliography_audit(text):
    body,refs=text.split('## References\n',1)
    entries={int(n) for n in re.findall(r'^(\d+)\. ',refs,re.M)}
    # [0,1] is the physical domain interval, not a numbered citation.
    cited={int(n) for group in re.findall(r'\[(\d+(?:,\d+)*)\]',body)
           if all(int(n)>0 for n in group.split(',')) for n in group.split(',')}
    if entries!=cited: raise ValueError(f'Reference/citation mismatch: {entries}, {cited}')
    return len(entries)


def build_paper(path):
    text=(ROOT/'paper/manuscript.md').read_text();bibliography_audit(text)
    sty=styles();width=A4[0]-108
    captions={int(n):f'Figure {n}. {c}' for n,c in re.findall(r'\*\*Figure (\d+)\.\*\* (.+)',text)}
    abstract=text.split('## Abstract\n\n')[1].split('\n## 1.')[0]
    story=[Spacer(1,14),Paragraph('InverPINN',ParagraphStyle('brand',fontName='Sans-Bold',fontSize=31,leading=36,textColor=colors.HexColor(INK))),
      Spacer(1,10),Paragraph(TITLE.split(': ',1)[1],ParagraphStyle('title',fontName='Sans-Bold',fontSize=19,leading=25,textColor=colors.HexColor(INK))),
      Spacer(1,18),Paragraph(AUTHOR,ParagraphStyle('author',fontName='Sans',fontSize=12,leading=16)),
      Spacer(1,17),Paragraph('Abstract',sty['h1']),Paragraph(rich(abstract),sty['body']),Spacer(1,6),figure(1,captions[1],width,sty),PageBreak()]
    main='## 1. Introduction'+text.split('## 1. Introduction',1)[1].split('## Main figures',1)[0]
    eq=0
    for block in re.split(r'\n\s*\n',main.strip()):
        block=block.strip()
        if not block:continue
        if block.startswith('\\['):
            eq+=1
            equation=Table([[Paragraph(EQUATIONS[eq-1],sty['equation']),Paragraph(f'({eq})',sty['equation'])]],colWidths=[width-28,28])
            equation.setStyle(TableStyle([('VALIGN',(0,0),(-1,-1),'MIDDLE'),('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),8)]))
            story.append(equation)
        elif block.startswith('### '):
            if block.startswith('### 2.3 '):story.append(figure(2,captions[2],width,sty))
            story.append(Paragraph(rich(block[4:]),sty['h2']))
        elif block.startswith('## '):story.append(Paragraph(rich(block[3:]),sty['h1']))
        elif block.startswith('|'):
            story.extend([Paragraph('Table 3. Final blind benchmark; all 30 cases per method.',sty['caption']),table_from_md(block,[68,65,79,82,90,width-384],sty['table']),Spacer(1,10)])
        elif re.match(r'\d+\. ',block):
            for line in block.splitlines():story.append(Paragraph(rich(line),sty['body']))
        else:story.append(Paragraph(rich(block.replace('\n',' ')),sty['body']))
    if eq!=len(EQUATIONS):raise ValueError('Manuscript equations changed; review typesetting explicitly')
    story.append(Paragraph('References',sty['h1']))
    for line in text.split('## References\n',1)[1].strip().split('\n'):
        if line.strip():story.append(Paragraph(rich(line),sty['caption']))
    story.extend([PageBreak(),Paragraph('Main figures',sty['h1'])])
    for number in range(3,9):
        if number in (5,7):story.append(PageBreak())
        story.append(figure(number,captions[number],width,sty))
    story.extend([PageBreak(),Paragraph('Appendix A. Parameter and method tables',sty['h1'])])
    for number,name,widths in [(1,'table1_physics',[130,185,width-315]),(2,'table2_methods',[74,137,137,width-348])]:
        story.extend([Paragraph(f'Table {number}. '+('Physical and numerical parameters.' if number==1 else 'Frozen inverse-method configurations.'),sty['caption']),
          table_from_md((ROOT/f'paper/generated/tables/{name}.md').read_text(),widths,sty['table']),Spacer(1,18)])
    story.append(KeepTogether([Paragraph('Table 4. Assumptions and limitations.',sty['caption']),table_from_md((ROOT/'paper/generated/tables/table4_limits.md').read_text(),[210,width-210],sty['table'])]))
    def footer(canvas,doc):
        canvas.setFont('Sans',8);canvas.setFillColor(colors.HexColor('#61717B'))
        canvas.drawString(54,28,'InverPINN | Controlled synthetic study');canvas.drawRightString(A4[0]-54,28,str(doc.page))
    doc=SimpleDocTemplate(str(path),pagesize=A4,rightMargin=54,leftMargin=54,topMargin=43,bottomMargin=47,title=TITLE,author=AUTHOR,subject='Final controlled research report; J1 failed confirmation')
    doc.build(story,onFirstPage=footer,onLaterPages=footer)


def build_poster(path,metrics):
    """A1 landscape, native text and two unchanged scientific figures."""
    w,h=2383.94,1683.78
    c=Canvas(str(path),pagesize=(w,h),pageCompression=1);c.setTitle('InverPINN Research Poster');c.setAuthor(AUTHOR)
    def para(text,x,top,width,size=27,leading=None,bold=False,color=INK):
        p=Paragraph(text,ParagraphStyle('poster',fontName='Sans-Bold' if bold else 'Sans',fontSize=size,leading=leading or size*1.3,textColor=colors.HexColor(color)))
        _,ph=p.wrap(width,h)
        if top+ph>h-42:raise ValueError('Poster text overflow')
        p.drawOn(c,x,h-top-ph)
    para('InverPINN',78,52,1300,78,bold=True)
    para('Source-strength bias in physics-informed pollution inversion',80,151,2220,42,bold=True)
    para(AUTHOR+' / Controlled synthetic research',82,218,2200,26,color=BLUE)
    para('Can sparse concentration sensors recover both the location and strength of a pollution source?',82,279,2200,34)
    left,right,col=82,1255,1045
    para('METHOD AND CONTROLLED DESIGN',left,365,col,31,bold=True,color=BLUE)
    para('A neural concentration field fits sparse observations and the transport equation. J1 analytically profiles source amplitude Q from the PDE residual.',left,420,col,29)
    para('C<sub>t</sub> + uC<sub>x</sub> + vC<sub>y</sub> = D(C<sub>xx</sub> + C<sub>yy</sub>) + QG',left,572,col,33,bold=True)
    para('One Gaussian (σ=0.08), known wind (0.35, -0.15), D=0.005. Unit-square domain. Zero initial and boundary concentration.',left,648,col,27)
    para('20 fixed sensors · 81 times · zero added noise<br/>641×641 reference · 30 untouched final scenarios',left,777,col,27)
    para('SOURCE-STRENGTH DIAGNOSIS',left,887,col,31,bold=True,color=BLUE)
    c.drawImage(str(ROOT/'paper/generated/figures/04_source_strength.png'),left,h-967-500,width=col,height=500,preserveAspectRatio=True,anchor='n')
    para('J1 underestimated Q in 28/30 cases (2 overestimates). The identity line shows exact recovery. Q is peak intensity, not integrated emissions.',left,1468,col,24)
    para('FINAL BLIND BENCHMARK',right,365,col,31,bold=True,color=BLUE)
    para('J1: 23/30',right,413,col,78,bold=True,color=RED)
    para('76.67% recovered; required 27/30 (90%)',right,521,col,31,bold=True,color=RED)
    para('FAILED preregistered confirmation',right,568,col,31,bold=True,color=RED)
    para('Classical inverse: 30/30',right,630,col,47,bold=True,color=BLUE)
    data=[['Median error','J1','Classical'],['Localization',metrics['J1']['localization'],metrics['classical']['localization']],['Relative Q',metrics['J1']['Q'],metrics['classical']['Q']],['Concentration L2',metrics['J1']['L2'],metrics['classical']['L2']]]
    ps=ParagraphStyle('poster-table',fontName='Sans',fontSize=25,leading=33)
    t=Table([[Paragraph(v,ps) for v in row] for row in data],colWidths=[col*.45,col*.26,col*.29])
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#E8F0F3')),('VALIGN',(0,0),(-1,-1),'TOP'),('TOPPADDING',(0,0),(-1,-1),9),('BOTTOMPADDING',(0,0),(-1,-1),9)]))
    _,th=t.wrap(col,300);t.drawOn(c,right,h-713-th)
    para('PHYSICAL VALIDITY DID NOT ENSURE RECOVERY',right,955,col,29,bold=True,color=BLUE)
    para('J1 retained zero negativity and exact IC/BC. All seven joint failures missed the Q criterion despite identical observations for both methods.',right,1006,col,26)
    para('WORST LOCALIZATION CASE (030)',right,1123,col,27,bold=True,color=BLUE)
    c.drawImage(str(ROOT/'paper/generated/figures/08_failure.png'),right,h-1175-325,width=col,height=325,preserveAspectRatio=True,anchor='n')
    para('Mechanically selected and retained. Source-strength error: 62.183%. Concentration L2: 69.724%.',right,1510,col,22)
    para('LIMITS  Known transport, one source and zero noise. No validated real Almaty emitter attribution, noisy-sensor robustness or general PINN superiority.',82,1590,2220,25,bold=True)
    c.save()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-root',type=Path,default=ROOT);p.add_argument('--font-dir',type=Path)
    args=p.parse_args();out=args.output_root.resolve()
    if out != ROOT and out.is_relative_to(ROOT) and not out.is_relative_to(ROOT/'artifacts'):
        raise ValueError('Use the repository root, an artifacts/ subdirectory, or an external scratch directory')
    targets=[out/'paper/InverPINN_Paper.pdf',out/'presentation/InverPINN_Poster.pdf',out/'presentation/pdf_build_provenance.json']
    if any(t.exists() for t in targets):raise FileExistsError('Refusing to overwrite delivery outputs; choose a new --output-root')
    metrics=verified_inputs();fonts=register_fonts(args.font_dir)
    for t in targets:t.parent.mkdir(parents=True,exist_ok=True)
    build_paper(targets[0]);build_poster(targets[1],metrics)
    provenance=dict(author=AUTHOR,source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),scientific_commit='eb18380f1488f5c679a0981da75e667fa3ba1b51',training_performed=False,simulation_performed=False,manuscript_sha256=sha(ROOT/'paper/manuscript.md'),fonts=fonts,packages={p:importlib.metadata.version(p) for p in ('reportlab','pypdf','Pillow')},metrics=metrics,files={str(t.relative_to(out)):sha(t) for t in targets[:2]})
    provenance['builder_sha256']=sha(__file__)
    targets[2].write_text(json.dumps(provenance,indent=2,sort_keys=True)+'\n')
    print('Built paper and A1 poster from preserved evidence. J1 confirmation remains FAILED.')


if __name__=='__main__':main()
