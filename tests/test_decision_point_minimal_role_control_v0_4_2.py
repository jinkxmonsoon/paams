import ast
import hashlib
import json
from dataclasses import replace
from pathlib import Path

from btom_v2.decision_point_content_matched_control_prompting_v0_4_0 import (
    CONDITIONS_BY_FAMILY, EXPLICIT_SELF_BELIEF, FIRST_ORDER_TITLE,
    MATCHED_DECISION_RECORD, MESSAGE_RECORD, PARTNER_BELIEF, SECOND_ORDER_TITLE,
    render_prompt as render_parent_prompt,
)
from btom_v2.decision_point_minimal_role_control_prompting_v0_4_2 import (
    REFERENCE_ROLE, ROLE_NORMALIZATION, TREATMENT_ROLE, audit_pair,
    render_all_prompts, render_prompt, run_audit,
)
from btom_v2.decision_point_natural_control_prompting_v0_3_0 import COMMON_SECTIONS
from btom_v2.decision_point_scenarios import CASES

ROOT=Path(__file__).parents[1]
MODULE=ROOT/'btom_v2/decision_point_minimal_role_control_prompting_v0_4_2.py'
MANIFEST=json.loads((ROOT/'btom_v2/decision_point_minimal_role_control_manifest_v0_4_2.json').read_text())


def _mutate(prompt,title,old,new):
    sections=tuple((name,text.replace(old,new) if name==title else text) for name,text in prompt.sections)
    return replace(prompt,sections=sections,prompt='\n\n'.join(text for _,text in sections))


def test_exact_roles_ascii_length_and_mapping():
    assert REFERENCE_ROLE=='record' and TREATMENT_ROLE=='belief'
    assert ROLE_NORMALIZATION=={'decision_record':'record','self_belief':'belief','message_record':'record','partner_belief':'belief'}
    for role in (REFERENCE_ROLE,TREATMENT_ROLE):
        assert role.isascii() and len(role)==6 and len(role.split())==1
    assert MANIFEST['role_normalization']==ROLE_NORMALIZATION


def test_conditions_and_population_unchanged():
    assert MANIFEST['conditions_by_family']=={k:list(v) for k,v in CONDITIONS_BY_FAMILY.items()}
    prompts=render_all_prompts()
    assert len(prompts)==16
    assert sum(p.family=='DP5' for p in prompts)==6
    assert sum(p.family=='DP7' for p in prompts)==10


def test_wrapper_delegates_and_only_role_values_change():
    source=MODULE.read_text()
    assert 'from .decision_point_content_matched_control_prompting_v0_4_0 import (' in source
    for case in CASES:
        for condition in CONDITIONS_BY_FAMILY[case.family]:
            parent=render_parent_prompt(case,condition); rendered=render_prompt(case,condition)
            assert tuple(n for n,_ in parent.sections)==tuple(n for n,_ in rendered.sections)
            for (name,a),(_,b) in zip(parent.sections,rendered.sections):
                for old_line,new_line in zip(a.splitlines(),b.splitlines()):
                    if old_line!=new_line:
                        assert old_line.startswith('representation_role="')
                        assert new_line.startswith('representation_role="')
                        assert ROLE_NORMALIZATION[old_line.split('"')[1]]==new_line.split('"')[1]


def test_all_primary_pairs_have_character_byte_and_single_field_parity():
    for case in CASES:
        ref=render_prompt(case,MATCHED_DECISION_RECORD); treatment=render_prompt(case,EXPLICIT_SELF_BELIEF)
        result=audit_pair(ref,treatment,case,FIRST_ORDER_TITLE)
        assert result['passed'] is True
        assert result['identical_character_count'] and result['identical_utf8_byte_count']
        if case.family=='DP7':
            ref=render_prompt(case,MESSAGE_RECORD); treatment=render_prompt(case,PARTNER_BELIEF)
            result=audit_pair(ref,treatment,case,SECOND_ORDER_TITLE)
            assert result['passed'] is True
            assert result['reference_role_is_record'] and result['treatment_role_is_belief']
            assert dict(ref.sections)[FIRST_ORDER_TITLE].endswith('representation_role="belief"')
            assert dict(treatment.sections)[FIRST_ORDER_TITLE].endswith('representation_role="belief"')


def test_global_audit_and_independent_H2_provenance():
    audit=run_audit()
    assert audit['prompt_count']==16 and audit['DP5_prompt_count']==6 and audit['DP7_prompt_count']==10
    assert audit['only_representation_role_changed_from_v0_4_0'] is True
    assert audit['common_sections_byte_identical_to_v0_4_0'] is True
    assert audit['observations_actions_messages_output_frozen'] is True
    assert audit['pairwise_audit_count']==6 and audit['pairwise_audits_passed'] is True
    assert audit['independent_H2_provenance_preserved'] is True
    assert audit['DP7_raw_evidence_exactly_once'] is True
    for key in ('hidden_state_leak_count','scoring_label_leak_count','condition_name_leak_count','source_message_field_count','legacy_marker_count'):
        assert audit[key]==0
    assert audit['tokenizer_counts_produced'] is False and audit['tokenizer_parity_claimed'] is False
    assert audit['model_or_api_execution'] is False and audit['real_execution_authorized'] is False
    for case in CASES:
        for condition in CONDITIONS_BY_FAMILY[case.family]:
            parent=render_parent_prompt(case,condition); rendered=render_prompt(case,condition)
            for name in COMMON_SECTIONS:
                assert dict(parent.sections)[name]==dict(rendered.sections)[name]


def test_negative_other_field_change_fails():
    case=CASES[0]
    ref=render_prompt(case,MATCHED_DECISION_RECORD); treatment=render_prompt(case,EXPLICIT_SELF_BELIEF)
    altered=_mutate(treatment,FIRST_ORDER_TITLE,'proposition="medical_kit_location"','proposition="other"')
    assert audit_pair(ref,altered,case,FIRST_ORDER_TITLE)['passed'] is False


def test_negative_duplicate_evidence_fails():
    case=next(c for c in CASES if c.family=='DP7')
    ref=render_prompt(case,MESSAGE_RECORD); treatment=render_prompt(case,PARTNER_BELIEF)
    duplicated=replace(treatment,prompt=treatment.prompt+'\n'+case.raw_delivered_messages[0])
    assert audit_pair(ref,duplicated,case,SECOND_ORDER_TITLE)['passed'] is False


def test_hashes_no_tokenizers_clients_workflow_or_execution():
    for path,digest in MANIFEST['immutable_parent_sha256'].items():
        assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==digest
    source=MODULE.read_text(); tree=ast.parse(source)
    imports={a.name for n in ast.walk(tree) if isinstance(n,(ast.Import,ast.ImportFrom)) for a in n.names}
    assert 'tiktoken' not in imports and 'openai_harmony' not in imports
    lower=source.lower()
    for bad in ('import groq','from groq','openai import','requests','httpx','urllib','api.groq.com','api.openai.com'):
        assert bad not in lower
    # Later-version workflows are allowed; this assertion is scoped to the
    # v0.4.2 design artifact.
    assert not (
        ROOT
        / ".github/workflows"
        / "decision_point_minimal_role_control_v0_4_2.yml"
    ).exists()
    assert MANIFEST['tokenizer_parity_claimed'] is False
    assert MANIFEST['tokenizer_execution_authorized'] is False
    assert MANIFEST['model_or_api_execution'] is False
    assert MANIFEST['real_execution_authorized'] is False


def test_no_workflow_assertion_is_version_scoped():
    source = Path(__file__).read_text()
    old_double_quoted = "glob(" + '"*minimal_role*"' + ")"
    old_single_quoted = "glob(" + "'*minimal_role*'" + ")"
    assert old_double_quoted not in source
    assert old_single_quoted not in source
    assert "decision_point_minimal_role_control_v0_4_2.yml" in source
    assert not (ROOT / ".github/workflows" / "decision_point_minimal_role_control_v0_4_2.yml").exists()
