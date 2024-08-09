import scipy
import numpy as np
from PIL import Image, ImageDraw
import cv2
def open_img(path):
    out = [[], [], []]
    with Image.open(path) as img:
        for row in range(0, img.height, 8):

            for column in range(0, img.width, 8):
                r, g, b = [], [], []
                for x in range(0, 8):
                    for y in range(0, 8):
                        pix = img.getpixel((column + x, row + y))
                        r.append(pix[0])
                        g.append(pix[1])
                        b.append(pix[2])
                out[0].append(scipy.fftpack.dct(np.array(r).reshape((8, 8))))
                out[1].append(scipy.fftpack.dct(np.array(g).reshape((8, 8))))
                out[2].append(scipy.fftpack.dct(np.array(b).reshape((8, 8))))
        return out

def code(out, mess, a):
    bits = txt_conv(mess + '\0')
    bits_i = 0
    for d in out[2]:
        max = 0
        ind = ()
        for i, e in enumerate(d):
            for ii, ee in enumerate(e):
                if np.abs(max) < np.abs(ee):
                    max = ee
                    ind = (i, ii)
        if bits[bits_i] == 0:
            b = -1
        else:
            b = 1
        d[ind[0]][ind[1]] += a * b
        bits_i += 1
        if bits_i >= len(bits):
            break
    if bits_i < len(bits):
        raise Exception('Места мало!!!')
    return out, bits

def decode(path_old, path_new):
    dct_old = open_img(path_old)
    dct_new = open_img(path_new)
    out_str = ''
    c = 0
    count = 0
    bit_arr = []
    for d_o, d_n in zip(dct_old[2], dct_new[2]):
        max = 0
        ind = ()
        for i, e in enumerate(d_o):
            for ii, ee in enumerate(e):
                if np.abs(max) < np.abs(ee):
                    max = ee
                    ind = (i, ii)
        if d_o[ind[0]][ind[1]] > d_n[ind[0]][ind[1]]:
            c <<= 1
            bit_arr.append(0)
        else:
            c = (c << 1) | 1
            bit_arr.append(1)
        count += 1
        if count == 8:
            if c == 0:
                break
            out_str += chr(c)
            c = 0
            count = 0
    # print(out_str)
    return out_str, bit_arr




def c_img(name, dct):
    dct_i = 0
    with Image.new('RGB', size=(768, 512)) as img:
        for row in range(0, img.height, 8):
            for column in range(0, img.width, 8):
                r, g, b = scipy.fftpack.idct(dct[0][dct_i])/16, scipy.fftpack.idct(dct[1][dct_i])/16, scipy.fftpack.idct(dct[2][dct_i])/16
                for x in range(0, 8):
                    for y in range(0, 8):
                        # pix = img.getpixel((column + x, row + y))
                        img.putpixel((column + x, row + y), (int(r[x][y]), int(g[x][y]), int(b[x][y])))
                dct_i += 1
        img.save(name)


def get_color(img, c):
    out = []
    for i, e in enumerate(img):
        out.append(np.zeros(e.shape[0]))
        for ii, ee in enumerate(e):
            out[i][ii] = ee[c]

    return np.array(out)

def PSNR(original, compressed):
    original = get_color(original, 0)
    compressed = get_color(compressed, 0)
    mse = np.mean((original - compressed) ** 2)
    if mse == 0:
        return 100
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

def txt_conv(messege):
    bits = []
    for line in messege:
        for sim in line:
            sim = ord(sim)
            for i in range(8):
                bits.append((sim & 0x1))
                sim = sim >> 1
            bits[len(bits) - 8:] = reversed(bits[len(bits) - 8:])
    return bits

def jpg(path):
    with Image.open(path) as img:
        im1 = img.copy()
        path_1 = 'temp.jpg'
        im1.save(path_1, quality=98)
    with Image.open(path_1) as img:
        im2 = img.copy()
        path_2 = 'temp_2.bmp'
        im2.save(path_2)

if __name__ == '__main__':
    s = 'MessdfsdfsdfsdfsdfsdfsdfsdfdsfsdkjMessdfsdfsdfsdfsdfsdfsdfsdfdsfsdkj'
    a = 200
    out = open_img('kodim03.bmp')
    out, bit_arr1 = code(out, s, a)
    c_img("cod_kodim03.bmp", out)
    # jpg('cod_kodim03.bmp')
    # out_old, bit_arr2 = decode('kodim03.bmp', 'temp_2.bmp')
    out_old, bit_arr2 = decode('kodim03.bmp', 'cod_kodim03.bmp')

    # print(bit_arr1)
    # print(bit_arr2)
    #
    # print(out_old)

    # img1 = cv2.imread('kodim03.bmp')
    # enc_img1 = cv2.imread('cod_kodim03.bmp', 1)
    # flag = 0
    #
    # for i, e in enumerate(bit_arr1):
    #     if i >= len(bit_arr2):
    #         bit_arr2.append('')
    #     if bit_arr1[i] != bit_arr2[i]:
    #         flag += 1
    # print(bit_arr1)
    # print(bit_arr2)
    # print(f'Размер сообщения = {len(s)}, Альфа = {a} \nКоличество не совпавших bit = {flag}')
    # print(f'Размер сообщения = {len(s)}, Альфа = {a} PSNR(blue) = {PSNR(img1, enc_img1)}')

#PSNR ПО синей компоненте, jpg по мягче(5-10 раз), сравнение ошибки в байт арр и зависимость от альфы.
