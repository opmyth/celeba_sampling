import argparse, torch

parser = argparse.ArgumentParser()
parser.add_argument('--rs_path', type=str, default='results_rs.pt')
parser.add_argument('--ula_path', type=str, default='results_ula.pt')
parser.add_argument('--mala_path', type=str, default='results_mala.pt')
parser.add_argument('--gmh_path', type=str, default='results_gmh.pt')
parser.add_argument('--output_path', type=str, default='results_stylegan.pt')
args = parser.parse_args()

rs = torch.load(args.rs_path, weights_only=False)
ula = torch.load(args.ula_path, weights_only=False)
mala = torch.load(args.mala_path, weights_only=False)
gmh = torch.load(args.gmh_path, weights_only=False)

torch.save({
    'stylegan': {
        'w2_values': {'ULA': ula['w2_values'], 'MALA': mala['w2_values'], 'G_MH': gmh['w2_values']},
        'w2_baseline': rs['w2_baseline'],
        'samples': {'RS': rs['samples'], 'ULA': ula['samples'], 'MALA': mala['samples'], 'G_MH': gmh['samples']},
        'avg_log_reward': {'RS': rs['avg_log_reward'], 'ULA': ula['avg_log_reward'], 'MALA': mala['avg_log_reward'], 'G_MH': gmh['avg_log_reward']},
        'diversity': {'RS': rs['diversity'], 'ULA': ula['diversity'], 'MALA': mala['diversity'], 'G_MH': gmh['diversity']},
        'diversity_trace_cov': {'RS': rs['diversity_trace_cov'], 'ULA': ula['diversity_trace_cov'], 'MALA': mala['diversity_trace_cov'], 'G_MH': gmh['diversity_trace_cov']},
        'male_fraction': {'RS': rs['male_fraction'], 'ULA': ula['male_fraction'], 'MALA': mala['male_fraction'], 'G_MH': gmh['male_fraction']},
        'accept_rates': {'MALA': mala.get('accept_rates'), 'G_MH': gmh.get('accept_rates')},
        'z_init': ula.get('z_init'),
        'z_final': {'ULA': ula.get('z_final'), 'MALA': mala.get('z_final'), 'G_MH': gmh.get('z_final')},
    }
}, args.output_path)

print(f'Merged results saved to {args.output_path}')
