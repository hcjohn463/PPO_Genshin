# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 09:45:04 2020

@author: pang
"""

import numpy as np
from PIL import ImageGrab
import cv2
import time
import grabscreen
import os

def self_blood_count(self_gray):
    self_blood = 0
    for self_bd_num in self_gray[0]:
        if self_bd_num == 175 or self_bd_num == 139:
            self_blood += 1
    return self_blood

def boss_blood_count(boss_gray):
    boss_blood = 0
    for boss_bd_num in boss_gray[0]:
        if boss_bd_num == 139:
            boss_blood += 1
    return boss_blood

wait_time = 1
L_t = 3

window_size = (435,175,1400,800)

boss_blood_window = (835,215,1085,216) #左 下 右 上
self_blood_window = (881,895,1038,896) #左 下 d右 上

for i in list(range(wait_time))[::-1]:
    print(i+1)
    time.sleep(1)

last_time = time.time()

while(True):

    #printscreen = np.array(ImageGrab.grab(bbox=(window_size)))
    #printscreen_numpy = np.array(printscreen_pil.getdata(),dtype='uint8')\
    #.reshape((printscreen_pil.size[1],printscreen_pil.size[0],3))
    #pil格式耗时太长
    
    boss_screen_gray = cv2.cvtColor(grabscreen.grab_screen(boss_blood_window),cv2.COLOR_BGR2GRAY)#灰度图像收集
    self_screen_gray = cv2.cvtColor(grabscreen.grab_screen(self_blood_window),cv2.COLOR_BGR2GRAY)#灰度图像收集
    # screen_reshape = cv2.resize(screen_gray,(96,86))

    window_screen = grabscreen.grab_screen(window_size)

    self_blood = self_blood_count(self_screen_gray)
    boss_blood = boss_blood_count(boss_screen_gray)
    
    print(f"自己血量: {self_blood}, boss血量: {boss_blood}")
    
    cv2.imshow('window1',window_screen)
    #cv2.imshow('window3',printscreen)
    #cv2.imshow('window2',screen_reshape)
    
    #测试时间用
    print('loop took {} seconds'.format(time.time()-last_time))
    last_time = time.time()
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break
cv2.waitKey()# 视频结束后，按任意键退出
cv2.destroyAllWindows()
