"""Merge per-stage result files into one results_stylegan.pt with a schema
that's identical across every experiment kind (classifier, joint-classifier,
imagereward) - no more per-experiment-type merge scripts or missing fields."""
import argparse, os, torch

from config import EXPERIMENTS


def _slug(s):
    return s.lower().replace(' ', '_')


parser = argparse.ArgumentParser()
parser.add_argument('--experiment', required=True, choices=list(EXPERIMENTS))
parser.add_argument('--prompt', type=str, default=None,
                     help='override the config default prompt (imagereward experiments only)')
parser.add_argument('--expr_dir', type=str, default=None)
parser.add_argument('--output_path', type=str, default=None)
args = parser.parse_args()

cfg = EXPERIMENTS[args.experiment]
prompt = args.prompt or cfg.prompt
# imagereward experiments nest by prompt so different prompts' runs never
# collide (bald_ir has 3) - classifier experiments are unaffected (no prompt).
expr_dir = args.expr_dir
if expr_dir is None:
    expr_dir = f'experiments/{args.experiment}'
    if cfg.kind == 'imagereward':
        expr_dir = os.path.join(expr_dir, f'prompt_{_slug(prompt)}')
output_path = args.output_path or f'{expr_dir}/results_stylegan.pt'


def _load(stage_name):
    path = os.path.join(expr_dir, f'results_{stage_name}.pt')
    # map_location='cpu': stage files contain CUDA tensors (saved from GPU
    # memory by the sampler jobs), but merging is CPU-only work and may run
    # on a GPU-less allocation (submit_merge.sh) - without this, torch.load
    # crashes wherever torch.cuda.is_available() is False. Also means the
    # merged results_stylegan.pt holds CPU tensors, which is what every
    # consumer (notebook, EXPERIMENTS.md stats) wants anyway.
    return torch.load(path, weights_only=False, map_location='cpu') if os.path.exists(path) else None


prior = _load('prior')
rs = _load('rs')
sampler_results = {s: _load(s.lower()) for s in cfg.samplers}
stages = {'Prior': prior, 'RS': rs, **sampler_results}


def _field(key):
    # only include a stage if it actually has this key (e.g. RS has no
    # 'w2_values' - it's the reference other samplers are compared against,
    # not a None placeholder standing in for a missing value)
    return {name: data[key] for name, data in stages.items() if data is not None and key in data}


merged = {
    args.experiment: {
        'w2_values': _field('w2_values'),         # Prior + each sampler (RS has none - it's the reference)
        'w2_baseline': rs['w2_baseline'] if rs else None,  # RS self-W2 (lower bound)
        'samples': _field('samples'),
        'avg_log_reward': _field('avg_log_reward'),
        'diversity': _field('diversity'),
        'diversity_trace_cov': _field('diversity_trace_cov'),
        'male_fraction': _field('male_fraction'),
        'accept_rates': {name: data['accept_rates'] for name, data in sampler_results.items()
                          if data is not None and data.get('accept_rates') is not None},
        'rs_accept_rate': rs.get('accept_rate') if rs else None,
        'rs_r_max': rs.get('r_max') if rs else None,          # None for classifier experiments
        'prompt': rs.get('prompt') if rs else None,           # None for classifier experiments
        'z_init': next((d.get('z_init') for d in sampler_results.values()
                         if d and d.get('z_init') is not None), None),
        'z_final': {name: data.get('z_final') for name, data in sampler_results.items() if data},
    }
}

os.makedirs(expr_dir, exist_ok=True)
torch.save(merged, output_path)
print(f'Merged results saved to {output_path}')
