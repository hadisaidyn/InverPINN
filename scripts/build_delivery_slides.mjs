/** Build an editable 10-slide research deck from preserved evidence only.
 * Set ARTIFACT_TOOL_ENTRY, PRESENTATION_SKILL_DIR and RUNTIME_PYTHON to your
 * installed authoring runtime. No model, scientific solver or training runs.
 */
import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath, pathToFileURL} from 'node:url';
import {execFileSync} from 'node:child_process';
const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const [outArg]=process.argv.slice(2);
const out=path.resolve(outArg || path.join(root,'artifacts/final_delivery/slides_01'));
if(!out.startsWith(path.join(root,'artifacts')+path.sep)) throw new Error('Choose a new artifacts/ subdirectory for slide staging');
try {await fs.access(out); throw new Error('Refusing to overwrite an existing slide build');}
catch(error) {if(error.code!=='ENOENT') throw error;}
const skill=process.env.PRESENTATION_SKILL_DIR;
const python=process.env.RUNTIME_PYTHON;
if(!skill || !python) throw new Error('Set PRESENTATION_SKILL_DIR and RUNTIME_PYTHON');
const moduleName=process.env.ARTIFACT_TOOL_ENTRY || '@oai/artifact-tool';
const {Presentation,PresentationFile}=await import(path.isAbsolute(moduleName)?pathToFileURL(moduleName).href:moduleName);
const {finalizePresentation}=await import(pathToFileURL(path.join(skill,'container_tools/artifact_tool_utils.mjs')).href);
const content=JSON.parse(await fs.readFile(path.join(root,'presentation/content.json'),'utf8'));
const summary=JSON.parse(await fs.readFile(path.join(root,'paper/generated/summary.json'),'utf8'));
const font='DejaVu Sans', ink='#192D3D', blue='#16688A', red='#A63F26';
const deck=Presentation.create({slideSize:{width:1280,height:720}});
await fs.mkdir(out,{recursive:true});
const final=path.join(out,'final/InverPINN_Presentation.pptx');
await fs.mkdir(path.dirname(final),{recursive:true});
await fs.mkdir(path.join(out,'previews'),{recursive:true});
function text(slide,value,x,y,w,h,size=30,color=ink,bold=false){
  const shape=slide.shapes.add({geometry:'textbox',position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});
  shape.text=value;
  shape.text.style={typeface:font,fontSize:size,color,bold,autoFit:'none'};
  return shape;
}
async function image(slide,name,x,y,w,h){
  slide.images.add({blob:new Uint8Array(await fs.readFile(path.join(root,name))),contentType:'image/png',alt:name,fit:'contain',position:{left:x,top:y,width:w,height:h}});
}
let notes='# InverPINN speaker notes\n\nKhadis Aidyn. Target talk: 5-7 minutes.\n\n';
for(let index=0;index<content.slides.length;index++){
  const d=content.slides[index],n=index+1,slide=deck.slides.add();
  slide.background.fill='#FFFFFF';
  if(n===1){
    text(slide,'InverPINN',64,92,1130,100,76,ink,true);
    text(slide,content.subtitle,68,218,1090,110,44,blue,true);
    text(slide,d.body,68,386,1060,120,30);
    text(slide,content.author,68,590,1100,44,25);
  }else{
    text(slide,d.title,64,42,1152,105,44,ink,true);
    if(n===9){
      text(slide,d.body,68,157,1140,105,34,red,true);
      text(slide,'Error columns report medians across all 30 fresh scenarios',68,262,1140,30,20);
      const rows=[['Method','Recovery','Localization','Q error','Field L2']];
      for(const [key,label] of [['J1','J1 PINN'],['B_revised','B_revised'],['classical','Classical']]){
        const m=summary.methods[key];
        rows.push([label,`${m.recovered}/30`,m.metrics.localization_error.median.toFixed(6),`${(100*m.metrics.relative_strength_error.median).toFixed(3)}%`,`${(100*m.metrics.relative_l2.median).toFixed(3)}%`]);
      }
      const t=slide.tables.add({rows:4,columns:5,left:68,top:296,width:1140,height:256,columnWidths:[220,175,260,220,265],values:rows});
      t.borders.assign({fill:'#D7E2E7',width:0.6});
      for(let r=0;r<4;r++){
        t.rows[r].height=64;
        for(let c=0;c<5;c++){
          const cell=t.getCell(r,c);cell.fill=r===0?'#E8F0F3':'#FFFFFF';
          cell.text.style={typeface:font,fontSize:25,color:ink,bold:r===0};
        }
      }
      text(slide,'Joint recovery: localization ≤ 0.08 AND relative Q error ≤ 20%',68,590,1140,60,23);
    }else if(d.figure){
      text(slide,d.body,68,151,1140,n===4||n===10?150:112,n===4||n===10?27:29);
      const top=n===4||n===10?316:276;
      await image(slide,d.figure,60,top,1160,650-top);
    }else if(n===3 || n===8){
      text(slide,d.body,68,184,1130,385,n===3?34:32);
      text(slide,d.key_number,68,607,1120,56,23,blue);
    }else{
      text(slide,d.body,68,197,1080,330,36);
      text(slide,d.key_number,68,591,1130,56,24,blue);
    }
    text(slide,String(n),1170,671,50,27,17,'#61717B');
  }
  const note=`What to say: ${d.say}\n\nKey number: ${d.key_number}\n\nTransition: ${d.transition}\n\nSources: ${d.sources.join('; ')}`;
  slide.speakerNotes.textFrame.setText(note);
  notes+=`## Slide ${n}: ${d.title}\n\n${d.say}\n\n**Key number:** ${d.key_number}\n\n**Transition:** ${d.transition}\n\nSources: ${d.sources.map(s=>'`'+s+'`').join(', ')}.\n\n`;
}
const draft=path.join(out,'candidate.pptx');
const raw=path.join(out,'export.pptx');
await (await PresentationFile.exportPptx(deck)).save(raw);
// Artifact Tool exports generic creator metadata. Change only core properties,
// leaving every slide, note, relationship and embedded image byte-identical.
execFileSync(python,['-c',`
import sys, zipfile, xml.etree.ElementTree as ET
src, dst, author = sys.argv[1:]
ns = {'dc':'http://purl.org/dc/elements/1.1/', 'cp':'http://schemas.openxmlformats.org/package/2006/metadata/core-properties'}
with zipfile.ZipFile(src) as z, zipfile.ZipFile(dst,'x',compression=zipfile.ZIP_DEFLATED) as w:
    for info in z.infolist():
        data=z.read(info.filename)
        if info.filename=='docProps/core.xml':
            tree=ET.fromstring(data)
            for tag,value in [('dc:creator',author),('dc:title','InverPINN Research Presentation'),('cp:lastModifiedBy',author)]:
                prefix,name=tag.split(':'); key='{'+ns[prefix]+'}'+name
                node=tree.find(key)
                if node is None: node=ET.SubElement(tree,key)
                node.text=value
            data=ET.tostring(tree,encoding='utf-8',xml_declaration=True)
        w.writestr(info,data)
`,raw,draft,content.author]);
await finalizePresentation({workspaceDir:root,candidatePath:draft,finalPath:final,pythonExecutable:python,
 integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),
 layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),
 layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit','--require-native-table-slide','9'],
 explicitTotalSlideCount:10,requiredNativeTableOwnerSlides:[9],requiredNativeChartOwnerSlides:[],
 fontPolicy:{basis:'design',families:[font]},verifyArtifactToolImport:true,
 receiptPath:path.join(out,'validation.json')});
for(let i=0;i<deck.slides.items.length;i++){
  const slide=deck.slides.items[i];
  const preview=await deck.export({slide,format:'png',scale:1.5});
  await fs.writeFile(path.join(out,'previews',`slide-${String(i+1).padStart(2,'0')}.png`),new Uint8Array(await preview.arrayBuffer()));
}
await fs.writeFile(path.join(out,'speaker_notes.md'),notes.trimEnd()+'\n');
console.log(`Built ${content.slides.length} slides: ${final}`);
