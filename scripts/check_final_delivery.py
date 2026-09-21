"""Offline structural/number audit of presentation artifacts; never run science."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET
import zipfile

from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
AUTHOR = 'Khadis Aidyn'
HEADLINES = ('23/30', '27/30', '30/30', '76.67%', '90%',
             '0.005574', '5.975%', '7.423%', '0.001111', '1.218%', '1.507%')
PDFS = {'paper/InverPINN_Paper.pdf':14,
        'presentation/InverPINN_Poster.pdf':1,
        'presentation/InverPINN_Presentation.pdf':10}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_headlines(text):
    """Compare frozen headline strings, not a claim of exhaustive semantic QA."""
    missing = [value for value in HEADLINES if value not in text]
    if missing: raise ValueError(f'Missing or changed headline values: {missing}')
    if not re.search(r'fail', text, re.I):
        raise ValueError('Failed confirmation must be explicit')


def application_counts(text):
    result = {}
    for count in (50,100,150,250):
        section = re.search(rf'^## {count}-word[^\n]*\n+(.*?)(?=\n## |\Z)', text, re.M|re.S)
        if not section: raise ValueError(f'Missing {count}-word section')
        actual = len(section[1].strip().split())
        if actual != count: raise ValueError(f'{count}-word description has {actual} words')
        result[str(count)] = actual
    activity = re.search(r'^## Common App[^\n]*\n+(.+)',text,re.M)
    if not activity: raise ValueError('Missing Common App activity')
    actual = len(activity[1])
    if actual > 150: raise ValueError('Common App activity exceeds 150 characters')
    result['common_app_characters'] = actual
    return result


def check(root=ROOT):
    """Verify final files and sealed inputs without modifying them."""
    reports = {}
    for name, count in PDFS.items():
        path=root/name; reader=PdfReader(path)
        if len(reader.pages)!=count: raise ValueError(f'Unexpected page count: {name}')
        if reader.metadata.author != AUTHOR: raise ValueError(f'Incorrect author: {name}')
        texts=[p.extract_text() for p in reader.pages]
        if any(len(t.strip())<30 for t in texts): raise ValueError(f'Empty PDF page: {name}')
        full='\n'.join(texts); check_headlines(full)
        if any(s in full for s in ('/Users/', '/private/', 'file://', '```', '\\frac')):
            raise ValueError(f'Local path or unrendered markup: {name}')
        reports[name]={'pages':count,'bytes':path.stat().st_size,'sha256':sha(path),'author':reader.metadata.author}
        if name.startswith('paper/'):
            if 'Appendix A.' not in texts[12]: raise ValueError('Expected 12 main pages before appendix')
            for n in range(1,9):
                if f'Figure {n}.' not in full: raise ValueError(f'Missing Figure {n}')
            for n in range(1,5):
                if f'Table {n}.' not in full: raise ValueError(f'Missing Table {n}')
            for n in range(1,8):
                if f'({n})' not in full: raise ValueError(f'Missing equation number {n}')
    with zipfile.ZipFile(root/'presentation/InverPINN_Presentation.pptx') as z:
        if z.testzip(): raise ValueError('Corrupt PowerPoint package')
        slides=sorted(n for n in z.namelist() if re.fullmatch(r'ppt/slides/slide\d+.xml',n))
        notes=[n for n in z.namelist() if re.fullmatch(r'ppt/notesSlides/notesSlide\d+.xml',n)]
        if len(slides)!=10 or len(notes)!=10: raise ValueError('Ten slides and notes are required')
        a={'a':'http://schemas.openxmlformats.org/drawingml/2006/main'}
        full='\n'.join(' '.join(ET.fromstring(z.read(n)).itertext()) for n in slides)
        check_headlines(full)
        if AUTHOR not in full: raise ValueError('Confirmed author missing on cover')
        if not ET.fromstring(z.read('ppt/slides/slide9.xml')).findall('.//a:tbl',a):
            raise ValueError('Benchmark table must be editable')
        core=ET.fromstring(z.read('docProps/core.xml'))
        if core.find('{http://purl.org/dc/elements/1.1/}creator').text != AUTHOR:
            raise ValueError('PowerPoint author metadata mismatch')
        allowed={sha(p) for p in (root/'paper/generated/figures').glob('*.png')}
        media=[n for n in z.namelist() if n.startswith('ppt/media/')]
        if not media or any(hashlib.sha256(z.read(n)).hexdigest() not in allowed for n in media):
            raise ValueError('Slide image is not byte-identical canonical evidence')
        for name in z.namelist():
            if name.endswith('.rels') and b'TargetMode="External"' in z.read(name):
                raise ValueError('External relationship in presentation')
    path=root/'presentation/InverPINN_Presentation.pptx'
    reports[str(path.relative_to(root))]={'slides':10,'notes':10,'native_table_slide':9,'bytes':path.stat().st_size,'sha256':sha(path),'author':AUTHOR}
    evidence=json.loads((root/'paper/evidence/manifest.json').read_text())['files']
    for name,record in evidence.items():
        if sha(root/'paper/evidence'/name)!=record['sha256']:raise ValueError(f'Changed evidence: {name}')
    canonical=json.loads((root/'paper/generated/artifact_hashes.json').read_text())
    for name,digest in canonical.items():
        if sha(root/'paper/generated'/name)!=digest:raise ValueError(f'Changed canonical artifact: {name}')
    return {'artifacts':reports,'application_counts':application_counts((root/'docs/application_description.md').read_text()),
            'evidence_files_verified':len(evidence),'canonical_files_verified':len(canonical),
            'training_performed':False,'simulation_performed':False,
            'scope':'Structural/hash/headline checks; visual and scientific-claim review are separately documented.'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write-report',type=Path)
    args=parser.parse_args(); result=check(); serialized=json.dumps(result,indent=2,sort_keys=True)+'\n'
    if args.write_report:
        with args.write_report.open('x') as f:f.write(serialized)
    print(serialized)


if __name__=='__main__': main()
