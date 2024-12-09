# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 10:37:50 2020

@author: pang
"""

import ctypes
import time
from ctypes import wintypes


SendInput = ctypes.windll.user32.SendInput


W = 0x11
A = 0x1E
S = 0x1F
D = 0x20

Q = 0x10
E = 0x12

F = 0x21
esc = 0x01


# C struct redefinitions 
PUL = ctypes.POINTER(ctypes.c_ulong)

class KeyBdInput(ctypes.Structure):
    _fields_ = [
        ("wVk", ctypes.c_ushort),       # 虛擬鍵碼
        ("wScan", ctypes.c_ushort),     # 硬體掃描碼
        ("dwFlags", ctypes.c_ulong),    # 按鍵事件標誌
        ("time", ctypes.c_ulong),       # 事件時間戳
        ("dwExtraInfo", PUL),           # 附加資訊
    ]

# 定義硬體輸入結構
class HardwareInput(ctypes.Structure):
    _fields_ = [
        ("uMsg", ctypes.c_ulong),       # 訊息
        ("wParamL", ctypes.c_short),    # 參數 L
        ("wParamH", ctypes.c_ushort),   # 參數 H
    ]

# 定義滑鼠輸入結構
class MouseInput(ctypes.Structure):
    _fields_ = [
        ("dx", ctypes.c_long),          # 滑鼠 X 偏移
        ("dy", ctypes.c_long),          # 滑鼠 Y 偏移
        ("mouseData", ctypes.c_ulong),  # 滑鼠數據 (例如滾輪)
        ("dwFlags", ctypes.c_ulong),    # 滑鼠事件標誌
        ("time", ctypes.c_ulong),       # 事件時間戳
        ("dwExtraInfo", PUL),           # 附加資訊
    ]

# 定義輸入結構的聯合體
class Input_I(ctypes.Union):
    _fields_ = [
        ("ki", KeyBdInput),             # 鍵盤輸入
        ("mi", MouseInput),             # 滑鼠輸入
        ("hi", HardwareInput),          # 硬體輸入
    ]

# 定義完整的輸入結構
class Input(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_ulong),       # 輸入類型
        ("ii", Input_I),                # 輸入聯合
    ]

# Actuals Functions

def PressKey(hexKeyCode):
    extra = ctypes.c_ulong(0)
    ii_ = Input_I()
    ii_.ki = KeyBdInput( 0, hexKeyCode, 0x0008, 0, ctypes.pointer(extra) )
    x = Input( ctypes.c_ulong(1), ii_ )
    ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))

def ReleaseKey(hexKeyCode):
    extra = ctypes.c_ulong(0)
    ii_ = Input_I()
    ii_.ki = KeyBdInput( 0, hexKeyCode, 0x0008 | 0x0002, 0, ctypes.pointer(extra) )
    x = Input( ctypes.c_ulong(1), ii_ )
    ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))
    
def go_forward():
    PressKey(W)
    time.sleep(0.4)
    ReleaseKey(W)
    
def go_back():
    PressKey(S)
    time.sleep(0.4)
    ReleaseKey(S)
    
def go_left():
    PressKey(A)
    time.sleep(0.4)
    ReleaseKey(A)
    
def go_right():
    PressKey(D)
    time.sleep(0.4)
    ReleaseKey(D)
    
def skill():
    PressKey(E)
    time.sleep(0.2)
    ReleaseKey(E)

def burst():
    PressKey(Q)
    time.sleep(0.4)
    ReleaseKey(Q)

def press_esc():
    PressKey(esc)
    time.sleep(2)
    ReleaseKey(esc)

def start():
    PressKey(F)
    time.sleep(0.4)
    ReleaseKey(F)

# 模擬滑鼠按鍵操作的函數
def mouse_click(button="left"):
    extra = ctypes.c_ulong(0)
    ii_ = Input_I()
    
    # 左鍵或右鍵的按下和釋放標誌
    if button == "left":
        dwFlags_down = 0x0002  # MOUSEEVENTF_LEFTDOWN
        dwFlags_up = 0x0004    # MOUSEEVENTF_LEFTUP
    elif button == "right":
        dwFlags_down = 0x0008  # MOUSEEVENTF_RIGHTDOWN
        dwFlags_up = 0x0010    # MOUSEEVENTF_RIGHTUP
    else:
        raise ValueError("Invalid button! Use 'left' or 'right'.")
    
    # 模擬按下
    ii_.mi = MouseInput(0, 0, 0, dwFlags_down, 0, ctypes.pointer(extra))
    x = Input(ctypes.c_ulong(0), ii_)
    ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))
    
    # 模擬釋放
    ii_.mi = MouseInput(0, 0, 0, dwFlags_up, 0, ctypes.pointer(extra))
    x = Input(ctypes.c_ulong(0), ii_)
    ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))

    time.sleep(0.5)
def re():
    press_esc()
    mouse_click("left")
    time.sleep(5)
    mouse_click("left")
    time.sleep(3)
    mouse_click("left")
    for _ in range(4):
        go_forward()
        mouse_click("right")
    mouse_click("right")
    time.sleep(1)
    start()
if __name__ == '__main__':
    #前進突刺開e
    time.sleep(2)
    time1 = time.time()
    re()
    # while(1):
    #     go_forward() #向前
    #     mouse_click('left') #衝刺
    #     go_back()
    #需打開系統管理員