"""Inject one sampler's stage results into an experiment's EXISTING merged
results_stylegan.pt - for experiments whose per-stage files were cleaned up,
so merge_results.py can't rebuild the table from scratch (e.g. bald: only the
old merged file survives, per-stage results_{rs,prior,ula,mala,g_mh}.pt gone).

Adds the sampler's row to each per-sampler field dict WITHOUT touching the
existing rows, and preserves the file's existing top-level key (old
'stylegan'-keyed files stay 'stylegan'; current experiment-name-keyed files
stay as-is). This is the "add a column to the existing table" path, vs
merge_results.py's "rebuild the table from per-stage files" path.

Usage:
  python inject_sampler.py --experiment bald --sampler ANNEALED_MALA
  python inject_sampler.py --experiment bald_ir --sampler ANNEALED_MALA --prompt "a bald man"
"""
import argparse, os, torch
from config import EXPERIMENTS

# per-sampler field dicts keyed {sampler_name -> per-trial data}
PER_SAMPLER_FIELDS = ['samples', 'w2_values', 'avg_log_reward',
                      'diversity', 'diversity_trace_cov', 'male_fraction']

parser = argparse.ArgumentParser()
parser.add_argument('--experiment', required=True, choices=list(EXPERIMENTS))
parser.add_argument('--sampler', required=True, help='e.g. ANNEALED_MALA')
parser.add_argument('--prompt', default=None)
parser.add_argument('--merged_path', default=None)
parser.add_argument('--stage_path', default=None)
args = parser.parse_args()

cfg = EXPERIMENTS[args.experiment]
expr_dir = f'experiments/{args.experiment}'
if cfg.kind == 'imagereward':
    slug = (args.prompt or cfg.prompt).lower().replace(' ', '_')
    expr_dir = os.path.join(expr_dir, f'prompt_{slug}')
merged_path = args.merged_path or os.path.join(expr_dir, 'results_stylegan.pt')
stage_path = args.stage_path or os.path.join(expr_dir, f'results_{args.sampler.lower()}.pt')

merged = torch.load(merged_path, weights_only=False, map_location='cpu')
stage = torch.load(stage_path, weights_only=False, map_location='cpu')
top = list(merged)[0]                      # 'stylegan' (old) or experiment name (current)
d = merged[top]
print(f'merged {merged_path} top-key={top!r}, existing samplers={list(d["samples"].keys())}')

if args.sampler in d['samples']:
    print(f'WARNING: {args.sampler} already present - overwriting its row')

added = []
for field in PER_SAMPLER_FIELDS:
    if field in d and field in stage:
        d[field][args.sampler] = stage[field]
        added.append(field)
if stage.get('accept_rates') is not None:
    d.setdefault('accept_rates', {})[args.sampler] = stage['accept_rates']
    added.append('accept_rates')

torch.save(merged, merged_path)
print(f'Injected {args.sampler}: added to fields {added}')
print(f'samplers now: {list(d["samples"].keys())}')
