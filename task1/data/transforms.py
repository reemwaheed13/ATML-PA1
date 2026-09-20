import numpy as np
from PIL import Image
import torchvision.transforms as T
import torchvision.transforms.functional as TF

# normalization constants per backbone
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]   # openai clip weights
CLIP_STD  = [0.26862954, 0.26130258, 0.27577711]

# common 224 canvas — all interventions are built on this, per the manual
resize_224 = T.Resize((224, 224))


def normalize_for(name):
    # tensor + per-model normalize only (no resize/crop, already 224)
    mean, std = (CLIP_MEAN, CLIP_STD) if name == 'clip' else (IMAGENET_MEAN, IMAGENET_STD)
    return T.Compose([T.ToTensor(), T.Normalize(mean, std)])


def to_grayscale(img):
    return img.convert('L').convert('RGB')  # L = luminance only, back to RGB for model compat


def rotate_hue(img, factor=0.5):
    return TF.adjust_hue(img, hue_factor=factor)  # factor=0.5 → 180° shift (red→cyan etc.)


def translate(img, delta, direction):
    if delta == 0:
        return img
    dx, dy = {'right': (delta, 0), 'left': (-delta, 0),
               'up': (0, -delta), 'down': (0, delta)}[direction]
    t = TF.to_tensor(img)
    padded = TF.pad(t, padding=delta, padding_mode='reflect')  # mirror edges, no black borders
    x0, y0 = delta - dx, delta - dy                           # crop offset gives the shift
    w, h = img.size
    return TF.to_pil_image(padded[:, y0:y0+h, x0:x0+w])


def _perm_for(idx):
    """Deterministic non-identity 4×4 patch permutation for image index idx, seed 6304."""
    rng = np.random.default_rng(6304 + idx)
    p = rng.permutation(16)
    while np.all(p == np.arange(16)):
        p = rng.permutation(16)
    return p


def patch_shuffle(img, idx=None, grid=4):
    """
    idx: position of the image in the evaluation subset (0-indexed).
    Each idx gets a distinct permutation, fixed across all scripts and models.
    If idx is None falls back to idx=0 (should not happen in normal use).
    """
    perm = _perm_for(0 if idx is None else idx)
    arr = np.array(img)
    ph, pw = arr.shape[0] // grid, arr.shape[1] // grid
    patches = [arr[i*ph:(i+1)*ph, j*pw:(j+1)*pw]
               for i in range(grid) for j in range(grid)]
    result = np.zeros_like(arr)
    for dst, src in enumerate(perm):
        i, j = dst // grid, dst % grid
        result[i*ph:(i+1)*ph, j*pw:(j+1)*pw] = patches[src]
    return Image.fromarray(result)


if __name__ == "__main__":
    img = Image.fromarray(np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8))
    assert to_grayscale(img).size == (224, 224)
    assert rotate_hue(img).size == (224, 224)
    for d in [8, 16, 32]:
        for dr in ['right', 'left', 'up', 'down']:
            assert translate(img, d, dr).size == (224, 224)
    ps = patch_shuffle(img, idx=0)
    assert ps.size == (224, 224) and not np.array_equal(np.array(ps), np.array(img))
    # different indices must produce different permutations on the same image
    ps2 = patch_shuffle(img, idx=1)
    assert not np.array_equal(np.array(ps), np.array(ps2)), "idx 0 and 1 should differ"
    print("all transforms ok")
