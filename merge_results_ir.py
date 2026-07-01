import argparse, torch

parser = argparse.ArgumentParser()
parser.add_argument('--rs_path',     type=str, default='experiments/bald_ir/results_rs_ir.pt')
parser.add_argument('--mala_path',   type=str, default='experiments/bald_ir/results_mala_ir.pt')
parser.add_argument('--gmh_path',    type=str, default='experiments/bald_ir/results_gmh_ir.pt')
parser.add_argument('--output_path', type=str, default='experiments/bald_ir/results_merged_ir.pt')
args = parser.parse_args()

rs   = torch.load(args.rs_path,   weights_only=False)
mala = torch.load(args.mala_path, weights_only=False)
gmh  = torch.load(args.gmh_path,  weights_only=False)

torch.save({
    'bald_ir': {
        'w2_values': {
            'MALA': mala['w2_values'],
            'G_MH': gmh['w2_values'],
        },
        'samples': {
            'RS':   rs['samples'],
            'MALA': mala['samples'],
            'G_MH': gmh['samples'],
        },
        'avg_ir_score': {
            'RS':   rs.get('avg_ir_score'),
            'MALA': mala['avg_ir_score'],
            'G_MH': gmh['avg_ir_score'],
        },
        'accept_rates': {
            'MALA': mala['accept_rates'],
            'G_MH': gmh['accept_rates'],
        },
        'rs_accept_rate': rs['accept_rate'],
        'rs_r_max':       rs['r_max'],
        'prompt':         rs['prompt'],
    }
}, args.output_path)

print(f'Merged IR results saved to {args.output_path}')
