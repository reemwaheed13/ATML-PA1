import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF


def to_grayscale(img):
    return img.convert('L').convert('RGB')


def rotate_hue(img, factor=0.5):
    return TF.adjust_hue(img, hue_factor=factor)


def translate(img, delta, direction):
    if delta == 0:
        return img
    dx, dy = {'right': (delta, 0), 'left': (-delta, 0),
               'up': (0, -delta), 'down': (0, delta)}[direction]
    t = TF.to_tensor(img)
    padded = TF.pad(t, padding=delta, padding_mode='reflect')
    x0, y0 = delta - dx, delta - dy
    w, h = img.size
    return TF.to_pil_image(padded[:, y0:y0+h, x0:x0+w])


def _make_patch_perm(n=16, seed=6304):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    while np.all(perm == np.arange(n)):
        perm = rng.permutation(n)
    return perm

_PATCH_PERM = _make_patch_perm()


def patch_shuffle(img, grid=4):
    arr = np.array(img)
    H, W = arr.shape[:2]
    ph, pw = H // grid, W // grid
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
    assert ps.size == (224, 224)
    assert not np.array_equal(np.array(ps), np.array(img))

    print("all transforms ok")
