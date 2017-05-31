#!/usr/bin/env python
# encoding=utf-8

from PIL import Image, ImageDraw, ImageFont
import random 
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

"""
生成中文数字四则运算数据
c: chinese
m: math

"""

class ImageCaptcha(object):

    def __init__(self, size=(200, 60)):
        self.bgColor = self.random_color(200, 255)
        self.txt_num = u'零一二三四五六七八九十壹貳叄肆伍陸柒捌玖拾'
        self.txt_op = list(u'加减乘+-x') + [u"加上", u"减去", "乘以"]
        self.txt_eq = [u'等于', '等于几']
        self.image = Image.new('RGB', size, self.bgColor)

    def random_color(self, minc=0, maxc=255):
        return (random.randint(minc, maxc), 
                random.randint(minc, maxc),
                random.randint(minc, maxc))

    def gen_text(self, cnt=6):
        """
        随机生成验证码文本
        """
        a = random.choice(self.txt_num)
        b = random.choice(self.txt_op)
        c = random.choice(self.txt_num)
        d = random.choice(self.txt_eq)
        print a, b, c, d
        text = a+b+c+d
        return ''.join(text)

    def draw_text(self, pos, txt):
        draw = ImageDraw.Draw(self.image)
        fontSize = random.choice([22, 26, 28, 30])
        fontType = random.choice(['simsun.ttc']) #, 'Kaiti-SC-Bold.ttf', 'DrioidSans-Bold.ttf'])
        font = ImageFont.truetype(fontType, fontSize)
        fontColor = self.random_color(1, 180)
        draw.text(pos, txt, font=font, fill=fontColor)

    def rotate(self, angle):
        self.image = self.image.rotate(random.randint(-1*angle, angle), expand=0)
    
    def clear_bg(self):
        width, height = self.image.size
        for x in range(width):
            for y in range(height):
                pix = self.image.getpixel((x, y))
                if pix == (0, 0, 0):
                    self.image.putpixel((x,y), self.bgColor)
    
    def random_point(self):
        width, height = self.image.size
        x = random.randint(0, width)
        y = random.randint(0, height)
        return (x, y)
            
    def add_noise(self):
        start_point = self.random_point()
        end_point = self.random_point()
        draw = ImageDraw.Draw(self.image)
        for i in range(random.randint(2, 5)):
            draw.line((start_point, end_point), fill=self.random_color(), width=random.randint(0,2))
        
    def gen_captcha_image(self, text):
        for i, txt in enumerate(text):
            x = 2 + i * 30 + random.randint(-2, 2)
            y = random.randint(2, 10)
            self.draw_text((x, y), txt)
            self.rotate(3)
            self.add_noise()
        self.clear_bg()
        
        return self.image
 
def label_convert(label, detail=True):
    """
    结果转换
    """
    map_dict=dict((x, y) for x,y in zip(u'零一二三四五六七八九壹貳叄肆伍陸柒捌玖加减乘除+-x', u'0123456789123456789+-*/+-*'))
    map_dict[u'十'] = u'10'
    map_dict[u'拾'] = u'10'
    res = ''
    for c in label:
        if c in map_dict:
	        res += map_dict[c]
    if detail:
        res = eval(res)
    return str(res)   
 
if __name__ == '__main__':

    for i in range(10):
        test = ImageCaptcha()
        label = test.gen_text()
        img = test.gen_captcha_image(label)
        img.save('img/origin/%s_%d.jpg' % (label_convert(label), i))
	if i%5==0:
	    print i, 'done'

