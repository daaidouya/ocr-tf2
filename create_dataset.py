import os
import string
import random

from PIL import Image, ImageDraw, ImageFont
import numpy as np


def create_img():
    letters = string.digits + string.ascii_lowercase
    length = len(letters)
    im_50_blank = Image.new('RGB', (165, 32), (255, 255, 255))
    draw = ImageDraw.Draw(im_50_blank)
    num = ''
    for i in range(10):
        num += str(letters[random.randint(0, length - 1)])
    font = ImageFont.truetype('mvboli.ttf', 28)
    imwidth, imheight = im_50_blank.size
    font_width, font_height = draw.textsize(num, font)
    draw.text(
        ((imwidth - font_width - font.getoffset(num)[0]) / 2, (imheight - font_height - font.getoffset(num)[1]) / 2),
        text=num, font=font, fill=(0, 0, 0))
    return im_50_blank, num


f = open('./example/train1_annotations.txt', 'w+')
for i in range(1000):
    img, label = create_img()
    img.save(f'./example/Train1/{str(i)}.png')
    f.write(f'Train1/{str(i)}.png,{label}\n')