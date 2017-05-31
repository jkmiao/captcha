#!/usr/bin/env python
# encoding=utf-8

from PIL import Image, ImageDraw, ImageFont
import string
import random 

"""
生成中文数字四则运算数据
c: chinese
m: math

"""

class ImageCaptcha(object):

    def __init__(self, size=(200, 60)):
        self.imgSize = size
        self.bgColor = self.random_color(200, 255)
        self.txt_eq = string.digits+string.letters
        self.image = Image.new('RGB', size, self.bgColor)

    def random_color(self, minc=0, maxc=255):
        return (random.randint(minc, maxc), 
                random.randint(minc, maxc),
                random.randint(minc, maxc))

    def gen_text(self, cnt=6):
        # seq = '23456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        text = random.sample(self.txt_eq, 6)
        return ''.join(text)

    def draw_text(self, pos, txt):
        draw = ImageDraw.Draw(self.image)
        fontSize = random.choice([38, 40, 42])
#	fontType = random.choice(['Arimo-Bold.ttf', 'DroidSerif-Bold.ttf', 'DroidSans.ttf', 'DroidSerif-Italic.ttf', 'DroidSansMono.ttf', 'Arimo-Regular.ttf'])
        font = ImageFont.truetype('DroidSans-Bold.ttf', fontSize)
        fontColor = self.random_color(0, 160)
        draw.text(pos, txt, font=font, fill=fontColor)

    def rotate(self, angle):
	"""
	旋转
	"""
        self.image = self.image.rotate(random.randint(-1*angle, angle), expand=0)
    
    def clear_bg(self):
	"""
	旋转后的黑色背景替换
	"""
        width, height = self.image.size
        for x in range(width):
            for y in range(height):
                pix = self.image.getpixel((x, y))
                if pix == (0, 0, 0):
                    self.image.putpixel((x,y), self.bgColor)
    
    def random_point(self):
	"""
	生成随机点坐标
	"""
        width, height = self.image.size
        x = random.randint(0, width)
        y = random.randint(0, height)
        return (x, y)
            
    def add_noise(self):
	"""
	增加干扰线噪音
	"""
        start_point = self.random_point()
        end_point = self.random_point()
        draw = ImageDraw.Draw(self.image)
        for i in range(random.randint(2, 10)):
            draw.line((start_point, end_point), fill=self.random_color(), width=random.randint(0,2))
        
    def gen_captcha_image(self, text):
        for i, txt in enumerate(text):
            x = 5 + i * 33 + random.randint(-8, 8)
            y = random.randint(2, 10)
            self.draw_text((x, y), txt)
            self.rotate(3)
            self.add_noise()
        self.clear_bg()
        
        return self.image.resize(self.imgSize)
 
 
if __name__ == '__main__':

    for i in range(10):
        test = ImageCaptcha()
        label = test.gen_text()
        img = test.gen_captcha_image(label)
        img.save('img/origin/%s_%d.jpg' % (label.lower(), i))
	if i%5==0:
	    print i, 'done'

