from torchvision import transforms
from torchvision.datasets import ImageFolder

PACS_CLASSES = ['dog', 'elephant', 'giraffe', 'guitar', 'horse', 'house', 'person']
PACS_DOMAINS = ['art_painting', 'cartoon', 'photo', 'sketch']

_MEAN = [0.485, 0.456, 0.406]
_STD  = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(_MEAN, _STD),
])

eval_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(_MEAN, _STD),
])


def build_domain(root, domain, transform):
    if domain not in PACS_DOMAINS:
        raise ValueError(f"unknown PACS domain {domain!r}; expected one of {PACS_DOMAINS}")
    ds = ImageFolder(f"{root.rstrip('/')}/{domain}", transform=transform)
    if ds.classes != PACS_CLASSES:
        raise RuntimeError(
            f"{domain}: class layout {ds.classes} != expected {PACS_CLASSES}. "
            "Check the dataset directory structure."
        )
    return ds
