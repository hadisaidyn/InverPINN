"""Presentation-only tests: no model, checkpoint, training or simulator access."""
import ast
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT=Path(__file__).resolve().parents[1]


def module(name):
    spec=importlib.util.spec_from_file_location(name,ROOT/'scripts'/f'{name}.py')
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result)
    return result


CHECK=module('check_final_delivery')
BUILD=module('build_final_pdf')


def test_delivered_files_numbers_and_openability():
    report=CHECK.check()
    assert report['evidence_files_verified']==41
    assert report['training_performed'] is False
    assert report['artifacts']['paper/InverPINN_Paper.pdf']['pages']==14


@pytest.mark.parametrize('value', CHECK.HEADLINES)
def test_changed_headline_fails_loudly(value):
    text=' '.join(CHECK.HEADLINES)+' failed'
    with pytest.raises(ValueError,match='headline'):
        CHECK.check_headlines(text.replace(value,'WRONG'))


def test_failure_claim_cannot_disappear():
    with pytest.raises(ValueError,match='Failed confirmation'):
        CHECK.check_headlines(' '.join(CHECK.HEADLINES))


def test_word_and_character_limits():
    text=(ROOT/'docs/application_description.md').read_text()
    counts=CHECK.application_counts(text)
    assert counts=={'50':50,'100':100,'150':150,'250':250,'common_app_characters':132}
    with pytest.raises(ValueError,match='50-word'):
        CHECK.application_counts(text.replace('## 50-word description\n\n','## 50-word description\n\nEXTRA '))


def test_bibliography_does_not_confuse_domain_with_citation():
    assert BUILD.bibliography_audit((ROOT/'paper/manuscript.md').read_text())==6
    assert BUILD.bibliography_audit('Domain [0,1]. Citation [1].\n## References\n1. Existing reference')==1
    with pytest.raises(ValueError,match='mismatch'):
        BUILD.bibliography_audit('Citation [1,2].\n## References\n1. Existing reference')


def test_math_stars_are_not_markdown_emphasis():
    assert BUILD.rich('C*=0.88 and T*=0.8; max(0,Q*)')=='C*=0.88 and T*=0.8; max(0,Q*)'
    assert BUILD.rich('*Existing title*.')=='<i>Existing title</i>.'


def test_pdfs_cannot_be_silently_overwritten():
    result=subprocess.run([sys.executable,str(ROOT/'scripts/build_final_pdf.py')],capture_output=True,text=True)
    assert result.returncode!=0
    assert 'Refusing to overwrite' in result.stderr


@pytest.mark.parametrize('destination',['results','data','src','configs','paper/evidence'])
def test_pdf_builder_rejects_scientific_directories(destination):
    result=subprocess.run([sys.executable,str(ROOT/'scripts/build_final_pdf.py'),'--output-root',str(ROOT/destination)],capture_output=True,text=True)
    assert result.returncode!=0
    assert 'Use the repository root' in result.stderr


def test_frozen_scientific_paths_unchanged_since_polish():
    result=subprocess.run(['git','diff','--exit-code','1abae14e6cd4958aacc65cb2508f9a21d36feb5c','--',
        'src','configs','data','results','paper/evidence','paper/generated','paper/manuscript.md','paper/supplement.md'],
        cwd=ROOT,capture_output=True,text=True)
    assert result.returncode==0,result.stdout
    BUILD.verified_inputs()


def test_builders_have_no_scientific_imports():
    for name in ('build_final_pdf','check_final_delivery'):
        tree=ast.parse((ROOT/'scripts'/f'{name}.py').read_text())
        imports=[node.module or '' for node in ast.walk(tree) if isinstance(node,ast.ImportFrom)]
        imports += [alias.name for node in ast.walk(tree) if isinstance(node,ast.Import) for alias in node.names]
        assert not any(n.split('.')[0] in ('torch','inverpinn','scipy') for n in imports)


def test_notes_and_slide_content_match():
    content=json.loads((ROOT/'presentation/content.json').read_text())
    notes=(ROOT/'presentation/speaker_notes.md').read_text()
    assert len(content['slides'])==10
    assert content['author']=='Khadis Aidyn'
    spoken_words=sum(len(s['say'].split()) for s in content['slides'])
    assert 650<=spoken_words<=950
    for slide in content['slides']:
        assert slide['say'] in notes
        assert slide['key_number'] in notes
        assert slide['transition'] in notes
        for source in slide['sources']:assert (ROOT/source).is_file()


def test_delivery_packages_are_binary_for_git():
    files=list(CHECK.PDFS)+['presentation/InverPINN_Presentation.pptx']
    result=subprocess.run(['git','check-attr','text','diff','merge','--',*files],cwd=ROOT,capture_output=True,text=True,check=True)
    lines=result.stdout.splitlines()
    assert len(lines)==12
    assert all(line.endswith(': unset') for line in lines)
