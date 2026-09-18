"""Run a separately labelled attention reference; never launch S1 automatically."""
from copy import deepcopy

from .common import write_json
from .live_precision import run_live_preflight


def run_recovery(mod, natural, binding, cfg, report, out, prior):
    base = deepcopy(report)
    report.update(scope='cpu_attention_recovery_only_not_S1', status='running', stage='subset')
    write_json(out / 'diagnostic.json', report)
    subset = deepcopy(base)
    code = run_live_preflight(mod, natural, binding, cfg, subset, out / 'subset',
                             reference=True, audit_prior=prior, cpu_reference=True)
    report['subset'] = dict(status=subset['status'], report='subset/preflight-summary.json')
    if code:
        report.update(status='failed', stage='subset', full_preflight='not run')
        write_json(out / 'diagnostic.json', report)
        return code
    report['stage'] = 'full_preflight'
    write_json(out / 'diagnostic.json', report)
    full = deepcopy(base)
    code = run_live_preflight(mod, natural, binding, cfg, full, out / 'full',
                             reference=True, cpu_reference=True)
    report.update(status=full['status'], stage='finished',
                  full_preflight=dict(status=full['status'], report='full/preflight-summary.json'),
                  note='Separate reference preflight only. Original GPU failures remain; no S1 effect or speed claim.')
    write_json(out / 'diagnostic.json', report)
    return code
