import torch
import numpy as np
from utils import compute_w2, compute_diversity_cov, compute_avg_log_reward

rs   = torch.load('experiments/male_eye/results_rs.pt',   weights_only=False)
mala = torch.load('experiments/male_eye/results_mala.pt', weights_only=False)
gmh  = torch.load('experiments/male_eye/results_gmh.pt',  weights_only=False)

# W2: RS split in half; MALA/G_MH recomputed against new RS samples
w2_rs   = [compute_w2(t[:len(t)//2], t[len(t)//2:]) for t in rs['samples']]
w2_mala = [compute_w2(mala['samples'][i], rs['samples'][i]) for i in range(len(rs['samples']))]
w2_gmh  = [compute_w2(gmh['samples'][i],  rs['samples'][i]) for i in range(len(rs['samples']))]
print(f'RS   W2: {np.mean(w2_rs):.4f}±{np.std(w2_rs, ddof=1):.4f}')
print(f'MALA W2: {np.mean(w2_mala):.4f}±{np.std(w2_mala, ddof=1):.4f}')
print(f'G_MH W2: {np.mean(w2_gmh):.4f}±{np.std(w2_gmh, ddof=1):.4f}')

torch.save({'male_eye': {
    'w2_values': {
        'RS':   w2_rs,
        'MALA': w2_mala,
        'G_MH': w2_gmh,
    },
    'samples': {
        'RS':   rs['samples'],
        'MALA': mala['samples'],
        'G_MH': gmh['samples'],
    },
    'avg_log_reward': {
        'RS':   rs['avg_log_reward'],
        'MALA': mala['avg_log_reward'],
        'G_MH': gmh['avg_log_reward'],
    },
    'accept_rates': {
        'MALA': mala['accept_rates'],
        'G_MH': gmh['accept_rates'],
    },
    'rs_accept_rate': rs['accept_rate'],
}}, 'experiments/male_eye/results_merged.pt')
print('Merged saved to experiments/male_eye/results_merged.pt')
