import sys, os
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, 'stylegan2-ada-pytorch'))

import pickle
import torch
from models import classifier, StyleGAN2Wrapper

# Checkpoint filename exactly as downloaded from the StyleGAN2-ADA repo
_STYLEGAN_PKL = 'stylegan2_celeba.pkl'


class _NegatedClassifier(torch.nn.Module):
    """Wraps a classifier and negates its output logit, so
    logsigmoid(-logit) = log(1 - sigmoid(logit)) = log P(NOT this attribute).
    Used for a 'not_<attr>' name instead of a separately trained model -
    reuses the exact same checkpoint, just targets the opposite class."""
    def __init__(self, clf):
        super().__init__()
        self.clf = clf

    def forward(self, x):
        return -self.clf(x)


def load_classifier(name, device):
    # 'not_<attr>' reuses <attr>'s checkpoint, wrapped to negate the logit
    # (targets the opposite class - e.g. 'not_male' = female-presenting).
    base_name = name[len('not_'):] if name.startswith('not_') else name
    clf = classifier().to(device)
    clf.load_state_dict(torch.load(
        os.path.join(_HERE, 'clf_checkpoints', f'{base_name}_clf_aug.pth'),
        weights_only=False, map_location=device))
    clf.eval()
    clf.requires_grad_(False)
    if name.startswith('not_'):
        return _NegatedClassifier(clf).to(device).eval()
    return clf


def load_stylegan(device):
    ckpt_path = os.path.join(_HERE, 'stylegan2_checkpoints', _STYLEGAN_PKL)
    with open(ckpt_path, 'rb') as f:
        G = pickle.load(f)['G_ema'].to(device)
    stylegan = StyleGAN2Wrapper(G).to(device)
    stylegan.eval()
    stylegan.requires_grad_(False)
    return stylegan


def load_models(clf_names, device):
    """clf_names: the classifier(s) the experiment's posterior needs, e.g.
    ['smile'] or ['male', 'eyeglasses'] (joint posterior). Pass [] for
    ImageReward experiments that don't use a classifier posterior at all.

    male_clf is always loaded (deduplicated with clf_names if 'male' is
    already in there) since male_fraction is a cross-cutting diagnostic
    computed for every experiment regardless of its own posterior.

    Returns (stylegan, clfs: dict[name -> classifier], male_clf).
    """
    if isinstance(clf_names, str):
        clf_names = [clf_names]

    clfs = {name: load_classifier(name, device) for name in clf_names}
    male_clf = clfs['male'] if 'male' in clfs else load_classifier('male', device)
    stylegan = load_stylegan(device)

    return stylegan, clfs, male_clf
