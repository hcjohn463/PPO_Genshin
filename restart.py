# -*- coding: utf-8 -*-
"""
Created on Sat Jul  4 18:31:36 2020

@author: pang
"""

import directkeys
import time

def restart():
    print("角色死亡 Character Died")
    directkeys.re()
    print("重新自動開啟新的一輪 Automatically start a new round.")
  
if __name__ == "__main__":  
    restart()