from PIL import Image
import math
import sys

def val(want, default):
    if want is None:
        return default
    return want


class adapt:
    def __init__(self, darr, w, h, radius, fraction):
        self.darr = darr
        self.radius = radius
        self.h = h
        self.w = w
        self.fraction = fraction

    def __to_real(self, radian):
        x = self.w / 2 + self.radius * math.sin(radian)
        y = self.h / 2 - self.radius * math.cos(radian)
        x = round(x)
        y = round(y)

        x = max(0, min(self.w - 1, x))
        y = max(0, min(self.h - 1, y))

        return (x, y)

    def __getitem__(self, key):
        return self.darr[self.__to_real(key)]

    def __setitem__(self, key, value):
        key = self.__to_real(key)
        self.darr[key] = value

    def step(self):
        return 2*math.pi / self.fraction

def rot_ring(fake_arr_src, fake_arr_dst, n):
    dradian = fake_arr_src.step()

    src = 0
    dst = dradian * n

    if n < 0 or dst > 2*math.pi:
        return

    while src < 2*math.pi:
        fake_arr_dst[dst] = fake_arr_src[src]
        src += dradian
        dst += dradian

def ind_zad(img, n, frac):

    w = img.width
    h = img.height
    w = min(w, h)
    h = w

    res = img.copy()

    src = img.load()
    dst = res.load()

    r    = w // 2
    dist_r = r  

    for i in range(0, dist_r):
        rot_ring(adapt(src, w, h, r - i, frac), adapt(dst, w, h, r - i, frac), i * n)
    
    return res