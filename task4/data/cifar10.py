from torchvision import transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import Subset

CIFAR10_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']

# CIFAR-10 channel statistics
_MEAN = [0.4914, 0.4822, 0.4465]
_STD  = [0.2470, 0.2435, 0.2616]

# default recipe: crop + flip
default_train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(_MEAN, _STD),
])

# GCSC: exactly one change vs default -> RandAugment after crop+flip, before ToTensor
gcsc_train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.RandAugment(num_ops=2, magnitude=9),
    transforms.ToTensor(),
    transforms.Normalize(_MEAN, _STD),
])

# deterministic, unaugmented: used for val, test, unknowns, and Mahalanobis stats
eval_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(_MEAN, _STD),
])

TRAIN_TRANSFORMS = {'default': default_train_transform, 'gcsc': gcsc_train_transform}


def build_cifar10(root, train, transform, download=True):
    ds = CIFAR10(root=root, train=train, transform=transform, download=download)
    if ds.classes != CIFAR10_CLASSES:
        raise RuntimeError(
            f"CIFAR-10 class layout {ds.classes} != expected {CIFAR10_CLASSES}")
    return ds


def cifar10_train_val(root, splits, train_transform):
    # train split augmented for optimization; val split clean for selection
    train_full = build_cifar10(root, True, train_transform)
    val_full   = build_cifar10(root, True, eval_transform)
    return {
        'train': Subset(train_full, splits['train']),
        'val':   Subset(val_full,   splits['val']),
    }


def cifar10_eval_subset(root, splits, which):
    # train/val indices under the unaugmented eval transform (extraction, Mahalanobis)
    if which not in ('train', 'val'):
        raise ValueError(f"which must be 'train' or 'val', got {which!r}")
    return Subset(build_cifar10(root, True, eval_transform), splits[which])


def cifar10_test(root):
    return build_cifar10(root, False, eval_transform)
