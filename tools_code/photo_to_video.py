#!/usr/bin/env python
# -*-coding:utf-8-*-


#转换图片名字从0到n
import os
import math
def myrename(path):
    file_list = os.listdir(path)
    for i, fi in enumerate(file_list):
        old_dir = os.path.join(path, fi)
        filename = str(i) + "." + str(fi.split(".")[-1])
        new_dir = os.path.join(path, filename)
        try:
            os.rename(old_dir, new_dir)
        except Exception as e:
            print(e)
            print("Failed!")
        else:
            print("SUcess!")


if __name__ == "__main__":
    path = "D:/项目5003/code/hypernerf-main/hypernerf-main/ca2/hypernerf/curl/render/train"
    myrename(path)



