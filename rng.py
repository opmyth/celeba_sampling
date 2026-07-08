import torch


def make_generator(seed: int, device) -> torch.Generator:
    """Explicit per-stream RNG, used instead of global torch.manual_seed().

    Global manual_seed() is fragile when multiple independent chains/branches
    run in the same process: anything that consumes the default RNG between a
    reset and the draw you care about silently desyncs the stream. Passing an
    explicit generator into every randn/rand call removes that class of bug.
    """
    device = torch.device(device)
    try:
        g = torch.Generator(device=device)
    except RuntimeError:
        g = torch.Generator(device='cpu')
    g.manual_seed(seed)
    return g
