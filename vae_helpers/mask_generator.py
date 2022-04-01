"""
credit to/copied from CoModGAN's official implementation
"""
import numpy as np
from PIL import Image, ImageDraw
import math
import random


def fast_clip(ar, a, b):
    if isinstance(ar, np.ndarray):
        ar[np.less(ar, a)] = a
        ar[np.greater(ar, b)] = b
        return ar
    else:
        if ar < a:
            return a
        if ar > b:
            return b
        return ar

def RandomBrush(
        max_tries,
        s,
        min_num_vertex = 4,
        max_num_vertex = 18,
        mean_angle = 2*math.pi / 5,
        angle_range = 2*math.pi / 15,
        min_width = 12,
        max_width = 48,
        rng=None):
    if rng is None:
        rng = np.random.RandomState(np.random.randint(2**32-1))
    H, W = s, s
    average_radius = math.sqrt(H*H+W*W) / 8
    mask = Image.new('L', (W, H), 0)
    for _ in range(rng.randint(max_tries)):
        num_vertex = rng.randint(min_num_vertex, max_num_vertex)
        angle_min = mean_angle - rng.uniform(0, angle_range)
        angle_max = mean_angle + rng.uniform(0, angle_range)
        angles = rng.uniform(angle_min, angle_max, size=num_vertex)
        angles[::2] = 2*math.pi - angles[::2]

        h, w = mask.size
        vertex = [(int(rng.randint(0, w)), int(rng.randint(0, h)))]
        r = fast_clip(
                rng.normal(loc=average_radius, scale=average_radius//2, size=num_vertex),
                0, 2*average_radius)
        dx = r * np.cos(angles)
        dy = r * np.sin(angles)
        for i in range(num_vertex):
            new_x = fast_clip(vertex[-1][0] + dx[i], 0, w)
            new_y = fast_clip(vertex[-1][1] + dy[i], 0, h)
            vertex.append((int(new_x), int(new_y)))

        draw = ImageDraw.Draw(mask)
        width = int(rng.uniform(min_width, max_width))
        draw.line(vertex, fill=1, width=width)
        for v in vertex:
            draw.ellipse((v[0] - width//2,
                          v[1] - width//2,
                          v[0] + width//2,
                          v[1] + width//2),
                         fill=1)
        if rng.random() > 0.5:
            mask.transpose(Image.FLIP_LEFT_RIGHT)
        if rng.random() > 0.5:
            mask.transpose(Image.FLIP_TOP_BOTTOM)
    mask = np.asarray(mask, np.uint8)
    if rng.random() > 0.5:
        mask = np.flip(mask, 0)
    if rng.random() > 0.5:
        mask = np.flip(mask, 1)
    return mask

def RandomMask(s, hole_range=[0,1], rng=None):
    if rng is None:
        rng = np.random.RandomState(np.random.randint(2**32-1))
    coef = min(hole_range[0] + hole_range[1], 1.0)
    while True:
        mask = np.ones((s, s), np.uint8)
        def Fill(max_size):
            w, h = rng.randint(max_size), rng.randint(max_size)
            ww, hh = w // 2, h // 2
            x, y = rng.randint(-ww, s - w + ww), rng.randint(-hh, s - h + hh)
            mask[max(y, 0): min(y + h, s), max(x, 0): min(x + w, s)] = 0
        def MultiFill(max_tries, max_size):
            for _ in range(rng.randint(max_tries)):
                Fill(max_size)
        MultiFill(int(10 * coef), s // 2)
        MultiFill(int(5 * coef), s)
        mask = np.logical_and(mask, 1 - RandomBrush(int(20 * coef), s, rng=rng))
        hole_ratio = 1 - np.mean(mask)
        if hole_range is not None and (hole_ratio <= hole_range[0] or hole_ratio >= hole_range[1]):
            continue
        return mask[np.newaxis, ...].astype(np.float32)

def BatchRandomMask(batch_size, s, hole_ranges):
    return np.stack([RandomMask(s, hole_range=hole_range) for hole_range in hole_ranges], axis = 0)

def np_mask_generator(s, hole_range):
    def random_mask_generator(hole_range):
        while True:
            yield RandomMask(s, hole_range=hole_range)
    return iter(random_mask_generator(hole_range))


