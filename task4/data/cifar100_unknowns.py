from torchvision.datasets import CIFAR100
from torch.utils.data import Subset

from task4.data.cifar10 import eval_transform

# fixed unknown groups (CIFAR-100 fine class names), 800 test images each
NEAR_CLASSES = ['bus', 'pickup_truck', 'motorcycle', 'tractor',
                'wolf', 'fox', 'leopard', 'camel']
FAR_CLASSES  = ['bottle', 'bowl', 'chair', 'clock',
                'keyboard', 'mushroom', 'sunflower', 'wardrobe']

UNKNOWN_GROUPS = {'near': NEAR_CLASSES, 'far': FAR_CLASSES}


def build_cifar100_unknowns(root, group, download=True):
    if group not in UNKNOWN_GROUPS:
        raise ValueError(f"group must be one of {list(UNKNOWN_GROUPS)}, got {group!r}")
    names = UNKNOWN_GROUPS[group]
    # CIFAR-100 TEST only; eval transform matches the CIFAR-10 input pipeline
    ds = CIFAR100(root=root, train=False, transform=eval_transform, download=download)
    name_to_idx = {n: i for i, n in enumerate(ds.classes)}
    missing = [n for n in names if n not in name_to_idx]
    if missing:
        raise RuntimeError(f"CIFAR-100 fine classes not found: {missing}")
    wanted = {name_to_idx[n] for n in names}
    idx = [i for i, t in enumerate(ds.targets) if t in wanted]
    sub = Subset(ds, idx)
    if len(sub) != 800:
        raise RuntimeError(f"{group} unknowns: expected 800 images, got {len(sub)}")
    return sub
