import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF


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


# One fixed permutation for all images (seed 6304), computed once at import
rng = np.random.default_rng(6304)
_PATCH_PERM = rng.permutation(16)
while np.all(_PATCH_PERM == np.arange(16)):  # re-draw if identity (extremely unlikely)
    _PATCH_PERM = rng.permutation(16)


def patch_shuffle(img, grid=4):
    arr = np.array(img)
    ph, pw = arr.shape[0] // grid, arr.shape[1] // grid
    patches = [arr[i*ph:(i+1)*ph, j*pw:(j+1)*pw]
               for i in range(grid) for j in range(grid)]
    result = np.zeros_like(arr)
    for dst, src in enumerate(_PATCH_PERM):
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
    ps = patch_shuffle(img)
    assert ps.size == (224, 224) and not np.array_equal(np.array(ps), np.array(img))
    print("all transforms ok")
