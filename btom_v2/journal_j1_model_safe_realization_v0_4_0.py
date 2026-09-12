"""Selection-blind model-safe semantic realizations for Journal J1."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
FRAME=ROOT/'btom_v2/journal_j1_confirmatory_candidate_frame_v0_3_0.json';DEVELOPMENT=ROOT/'btom_v2/journal_j1_development_scenarios_v0_3_0.json'
TEMPLATES={'resource_location':('retrieval','staging','handoff','replenishment'),'tool_placement':('assembly','repair','calibration','inspection'),'rendezvous_destination':('pickup','transfer','regrouping','evacuation'),'delivery_destination':('supply delivery','component delivery','package handoff','equipment transfer'),'hazard_assignment':('inspection','containment','monitoring','clearance'),'maintenance_target':('preventive service','fault repair','calibration','replacement')}
DEVELOPMENT_TEMPLATES={'resource_location':('stock intake',),'tool_placement':('bench setup',),'rendezvous_destination':('crew assembly',),'delivery_destination':('parcel routing',),'hazard_assignment':('perimeter review',),'maintenance_target':('system upkeep',)}
FIRST=('Alden','Briar','Cora','Damon','Elara','Felix','Greta','Hugo','Iris','Jonas','Keira','Lumen','Mara','Nolan','Orla','Pavel','Quinn','Rhea','Silas','Talia','Una','Vero','Willa','Xenia','Yara','Zane')
NOUNS=('Cedar','Orion','Maple','Kestrel','Meridian','Juniper','Solace','Harbor','Willow','Falcon','Nimbus','Laurel','Comet','Birch','Atlas','Meadow','Saffron','Pioneer','Summit','Cobalt','Dawn','Ember','Fjord','Grove','Horizon','Ivory','Jasper','Lagoon','Mosaic','Nova')
PREFIX={'location':('Bay','Depot','Station','Storage'),'destination':('Hub','Gate','Dock','Terminal'),'sector':('Sector','Zone','Grid','Area'),'component':('Valve','Relay','Pump','Sensor')}
DEVELOPMENT_NAMES=('Amity','Basil','Celeste','Dorian','Esme','Florian','Giselle','Hadrian','Ines','Jovian','Katia','Leander','Minerva','Nestor','Opal','Percival','Rowan','Sabine')
def canonical(v):return json.dumps(v,sort_keys=True,separators=(',',':')).encode()
def safe_values(record,development=False):
 key=record['variant_id']+(' development' if development else ' candidate');base=int.from_bytes(hashlib.sha256(key.encode()).digest()[:8],'big');offset=700 if development else 0;kind=record['q_semantic_type'];prefixes=PREFIX[kind]
 values=[f'{"Annex " if development else ""}{prefixes[(base+offset+j)%4]} {NOUNS[(base//7+offset+j*7)%len(NOUNS)]} {FIRST[(base//13+offset+j*5)%len(FIRST)]}' for j in range(4)]
 return values
def realize(record,development=False):
 values=safe_values(record,development);family=(record['candidate_ordinal_in_cell']-1)%4+1 if not development else 1;context=(DEVELOPMENT_TEMPLATES if development else TEMPLATES)[record['archetype']][family-1];names=DEVELOPMENT_NAMES if development else FIRST;agent=names[int.from_bytes(hashlib.sha256((record['variant_id']+' agent '+str(development)).encode()).digest()[:2],'big')%len(names)]
 visible={'focal_agent_name':agent,'partner_name':names[(names.index(agent)+9)%len(names)],'task_context':context,'task_label':f'{context} operation','values':values,'true_task_value':values[0],'alternate_task_value':values[1],'true_context_value':values[2],'alternate_context_value':values[3],'static_context':([] if record['difficulty']=='direct' else [f'{"Annex " if development else ""}{NOUNS[(names.index(agent)+3)%len(NOUNS)]} schedule remains unchanged']),'relation_depth':record['q_relation_depth']}
 normalized={'archetype':record['archetype'],'difficulty':record['difficulty'],'template_family':family,'task_context':context,'relation_depth':record['q_relation_depth'],'has_static_context':bool(visible['static_context']),'parallel_chains':record['difficulty']=='compositional'}
 result={'variant_id':record['variant_id'],'archetype':record['archetype'],'difficulty':record['difficulty'],'template_family':family,'model_visible':visible,'normalized_template_signature_sha256':hashlib.sha256(canonical(normalized)).hexdigest(),'world_state_sha256':hashlib.sha256(canonical(visible)).hexdigest(),'structural_semantic_signature_sha256':record['semantic_signature_sha256']};result['realization_sha256']=hashlib.sha256(canonical(result)).hexdigest();return result
def realize_candidates(records):return [realize(r,False) for r in records]
def realize_development(records):return [realize(r,True) for r in records]
def load_structural_inputs():return json.loads(FRAME.read_text()),json.loads(DEVELOPMENT.read_text())
