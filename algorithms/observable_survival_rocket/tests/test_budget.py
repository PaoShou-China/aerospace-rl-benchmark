import ast
import json
from pathlib import Path
import pytest
from test_core import ROOT, patch


def test_real_source_preparation_and_additional_budget(tmp_path):
    source = ROOT.parent/'physical_belief_rocket_landing'
    if not (source/'scripts/train.py').exists():
        pytest.skip('real upstream checkout absent')
    bundle = patch.prepare(ROOT.parents[1], tmp_path/'prepared')
    cfg=json.loads((tmp_path/'prepared/smoke.json').read_text())
    assert cfg['additional_transitions']==256
    full=json.loads((tmp_path/'prepared/full.json').read_text())
    assert full['additional_transitions']==131072
    text=(bundle/'scripts/train.py').read_text()
    start=text.index('        if args.additional_transitions is not None:', text.index('if args.resume:'))
    end=text.index('        if remaining_transitions < 0',start)
    import textwrap
    from types import SimpleNamespace
    fragment=textwrap.dedent(text[start:end])
    env={'args':SimpleNamespace(additional_transitions=256),
         'status':{'new_environment_transitions':6225920},'transition_target':256}
    exec(fragment,env)
    assert env['remaining_transitions']==256
    assert env['transition_target']==6226176
    env={'args':SimpleNamespace(additional_transitions=None),
         'status':{'new_environment_transitions':1000},'transition_target':2000}
    exec(fragment,env)
    assert env['remaining_transitions']==1000
    assert "manifest['contact_parameters_identified'] = False" in text
    for file in bundle.rglob('*.py'):
        ast.parse(file.read_text(),filename=str(file))
