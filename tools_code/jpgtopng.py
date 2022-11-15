# python:批量实现jpg图片修改大小并转换为png

import os
import PIL.Image as Image

def changeJpgToPng(w, h,srcPath,dstPath):
    # 修改图像大小
    image = Image.open(srcPath)
    image = image.resize((w, h), Image.ANTIALIAS)

    # 将jpg转换为png
    png_name = str(dstPath)[0:-len('.jpg')] + '.png'
    image.save(png_name)

    print(png_name)

    # 去白底
    image = image.convert('RGBA')
    img_w, img_h = image.size
    color_white = (255, 255, 255, 255)
    for j in range(img_h):
        for i in range(img_w):
            pos = (i, j)
            color_now = image.getpixel(pos)
            if color_now == color_white:
                # 透明度置为0
                color_now = color_now[:-1] + (0,)
                image.putpixel(pos, color_now)
    image.save(png_name)


if __name__ == '__main__':
    t_w = 1080
    t_h = 1920
    srcPath = 'D:/colmap/images/'
    dstPath = 'D:/colmap/image2/'

    filename_list = os.listdir('D:/colmap/images/')
#    for d in dic:
#        if d.count('.jpg') > 0:
#            changeJpgToPng(t_w, t_h, 'image/' + d)

    for d in filename_list:
        if d.count('.jpg') > 0:
            changeJpgToPng(t_w, t_h,srcPath + d,dstPath + d)
        pass
    print("完成了...")






