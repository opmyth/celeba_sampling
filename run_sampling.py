import argparse
import torch
import numpy as np
import torch.nn.functional as F
import time


from tqdm import tqdm
from huggingface_hub import hf_hub_download
from models import CelebAGenerator, CelebaVAE, classifier
from samplers import latent_ULA_celeba, latent_MALA_celeba, latent_Gaussian_MH_celeba, rejection_sampling
from utils import compute_w2, compute_diversity, compute_male_fraction

parser = argparse.ArgumentParser()
parser.add_argument('--n_chains', type=int, default=100)
parser.add_argument('--n_steps', type=int, default=800)
parser.add_argument('--sigma', type=float, default=0.5)
parser.add_argument('--n_trials', type=int, default=10)
parser.add_argument('--dt', type=float, default=0.5)
parser.add_argument('--output_path', type=str, default='results.pt')

args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device {device}')


path_dcgan = hf_hub_download(
        repo_id="huggan/dcgan-celeba-faces", 
        filename="pytorch_model.bin"
)
path_vae = hf_hub_download(
        repo_id='hussamalafandi/VAE-CelebA',
        filename='vae_celeba_latent_200_epochs_10_batch_64_subset_80000.pth'
)

state_dict_dcgan = torch.load(path_dcgan, map_location=device, weights_only=False)
dcgan = CelebAGenerator()
dcgan.load_state_dict(state_dict_dcgan)
dcgan.to(device)
dcgan.eval()

state_dict_vae = torch.load(path_vae, map_location=device, weights_only=False)
vae = CelebaVAE()
vae.load_state_dict(state_dict_vae)
vae.to(device)
vae.eval()

smile_clf = classifier()
smile_clf.load_state_dict(torch.load('checkpoints/smile_clf.pth', weights_only=False))
smile_clf.to(device)
smile_clf.eval()

male_clf = classifier()
male_clf.load_state_dict(torch.load('checkpoints/male_clf.pth', weights_only=False))
male_clf.to(device)
male_clf.eval()



def run_trials(model, clf, male_clf, dt, sigma, n_chains, n_steps, device, n_trials=10, seed=321):
    torch.manual_seed(seed)
    
    w2_values = {'ULA': [], 'MALA': [], 'G_MH': []}
    w2_baseline = []

    samples = {'RS': None, 'ULA': None, 'MALA': None, 'G_MH': None}
    avg_log_reward = {'RS': [], 'ULA': [], 'MALA': [], 'G_MH': []}
    diversity = {'RS': [], 'ULA': [], 'MALA': [], 'G_MH': []}
    male_fraction = {'RS': [], 'ULA': [], 'MALA': [], 'G_MH': []}

    t = time.time()
    RS_samples = rejection_sampling(model, clf, n_chains * n_trials * 2, device=device)
    print(f"RS done: {time.time()-t:.2f}s", flush=True)

    t = time.time()
    chunks = torch.chunk(RS_samples, n_trials*2, dim=0)
    w2_baseline = [compute_w2(chunks[2*i], chunks[2*i + 1]) for i in range(n_trials)]
    samples['RS'] = list(chunks[::2])
    print(f"RS W2 baseline done: {time.time()-t:.2f}s", flush=True)

    t = time.time()
    ULA_samples = latent_ULA_celeba(model, clf, n_chains * n_trials, n_steps, dt, device=device)[-1]
    print(f"ULA done: {time.time()-t:.2f}s", flush=True)
    t = time.time()
    ULA_chunks = torch.chunk(ULA_samples, n_trials, dim=0)
    w2_values['ULA'] = [compute_w2(ULA_chunks[i], chunks[2*i]) for i in range(n_trials)]
    samples['ULA'] = list(ULA_chunks)
    print(f"ULA W2 done: {time.time()-t:.2f}s", flush=True)

    t = time.time()
    MALA_samples = latent_MALA_celeba(model, clf, n_chains * n_trials, n_steps, dt, device=device)[-1]
    print(f"MALA done: {time.time()-t:.2f}s", flush=True)
    t = time.time()
    MALA_chunks = torch.chunk(MALA_samples, n_trials, dim=0)
    w2_values['MALA'] = [compute_w2(MALA_chunks[i], chunks[2*i]) for i in range(n_trials)]
    samples['MALA'] = list(MALA_chunks)
    print(f"MALA W2 done: {time.time()-t:.2f}s", flush=True)

    t = time.time()
    gaussianMH_samples = latent_Gaussian_MH_celeba(model, clf, n_chains * n_trials, n_steps, sigma, device=device)[-1]
    print(f"G_MH done: {time.time()-t:.2f}s", flush=True)
    t = time.time()
    gaussianMH_chunks = torch.chunk(gaussianMH_samples, n_trials, dim=0)
    w2_values['G_MH'] = [compute_w2(gaussianMH_chunks[i], chunks[2*i]) for i in range(n_trials)]
    samples['G_MH'] = list(gaussianMH_chunks)
    print(f"G_MH W2 done: {time.time()-t:.2f}s", flush=True)

    t = time.time()
    for name in avg_log_reward:
        with torch.no_grad():
            avg_log_reward[name] = [F.logsigmoid(clf(model(samples[name][i]))).mean().item() for i in range(n_trials)]
    print(f"avg_log_reward done: {time.time()-t:.2f}s", flush=True)

    t = time.time()
    for name in diversity:
        diversity[name] = [compute_diversity(model, samples[name][i]).item() for i in range(n_trials)]
    print(f"diversity done: {time.time()-t:.2f}s", flush=True)

    t = time.time()
    for name in male_fraction:
        male_fraction[name] = [compute_male_fraction(model, male_clf, samples[name][i]) for i in range(n_trials)]
    print(f"male_fraction done: {time.time()-t:.2f}s", flush=True)

    return w2_values, w2_baseline, samples, avg_log_reward, diversity, male_fraction

start = time.time()
vae_w2_values, vae_w2_baseline, vae_samples, vae_avg_log_reward, vae_diversity, vae_male_fraction = run_trials(
    vae, smile_clf, male_clf, dt=args.dt, sigma=args.sigma, n_chains=args.n_chains, n_steps=args.n_steps, n_trials=args.n_trials, device=device
)

dcgan_w2_values, dcgan_w2_baseline, dcgan_samples, dcgan_avg_log_reward, dcgan_diversity, dcgan_male_fraction = run_trials(
    dcgan, smile_clf, male_clf, dt=args.dt, sigma=args.sigma, n_chains=args.n_chains, n_steps=args.n_steps, n_trials=args.n_trials, device=device
)
print(f"Total time: {time.time() - start:.2f}s")
torch.save({
    'vae': {
        'w2_values': vae_w2_values,
        'w2_baseline': vae_w2_baseline,
        'samples': vae_samples,
        'avg_log_reward': vae_avg_log_reward,
        'diversity': vae_diversity,
        'male_fraction': vae_male_fraction,
    },
    'dcgan': {
        'w2_values': dcgan_w2_values,
        'w2_baseline': dcgan_w2_baseline,
        'samples': dcgan_samples,
        'avg_log_reward': dcgan_avg_log_reward,
        'diversity': dcgan_diversity,
        'male_fraction': dcgan_male_fraction,
    }
}, args.output_path)

print(f'Results saved to {args.output_path}')
