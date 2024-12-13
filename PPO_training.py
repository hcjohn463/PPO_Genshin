# -*- coding: utf-8 -*-
import numpy as np
from grabscreen import grab_screen
import cv2
import time
import directkeys
from getkeys import key_check
from PPO_pytorch_gpu import PPO  # 替換為 PPO 類
import os
from restart import restart
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from ultralytics import YOLO
import threading
import queue
import pyautogui
from PIL import ImageGrab
import win32api
import win32con
from background import BackgroundObjectDetector



def pause_game(paused):
    keys = key_check()
    if 'J' in keys:
        if paused:
            paused = False
            print('start game')
            time.sleep(1)
        else:
            paused = True
            print('pause game')
            time.sleep(1)
    if paused:
        print('paused')
        while True:
            keys = key_check()
            if 'J' in keys:
                if paused:
                    paused = False
                    print('start game')
                    time.sleep(1)
                    break
                else:
                    paused = True
                    time.sleep(1)
    return paused


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


def take_action(action):
    if action == 0:  # w
        directkeys.go_forward()
    elif action == 1:  # a
        directkeys.go_left()
    elif action == 2:  # d
        directkeys.go_right()
    elif action == 3:  # normal attack
        directkeys.mouse_click('left')
    elif action == 4:  # elemental skill
        directkeys.skill()
    elif action == 5:  # elemental burst
        directkeys.burst()
    elif action == 6:  # dodge
        directkeys.mouse_click('right')

def action_judge(boss_blood, next_boss_blood, self_blood, next_self_blood, stop, emergence_break, e_in_cd, q_in_cd,
                 safe_count, combo_count, action, distance_to_boss=None):
    if next_self_blood < 1:  # self dead
        if emergence_break < 2:
            reward = -50
            done = 1
            stop = 0
            emergence_break += 1
            return reward, done, stop, emergence_break, safe_count, combo_count
        else:
            reward = -50
            done = 1
            stop = 0
            emergence_break = 100
            return reward, done, stop, emergence_break, safe_count, combo_count

    # 基礎行動獎勵
    action_reward = 0
    if action in [0, 1, 2, 6]:  # 移動
        action_reward = -2  # 輕微懲罰過多移動
    elif action == 3:  # 普通攻擊
        action_reward = 5  # 鼓勵進行攻擊

    # 技能懲罰機制
    skill_reward = 0
    if action == 4:  # E技能
        if (boss_blood - next_boss_blood) > 0:
            skill_reward += 20
        if e_in_cd:
            skill_reward = -20  # 嚴重懲罰在CD時使用E
    
    elif action == 5:  # Q技能
        if q_in_cd:
            skill_reward = -20 # 嚴重懲罰在CD時使用Q

    # 傷害獎勵（大幅提高）
    damage_reward = 0
    blood_diff = boss_blood - next_boss_blood
    if blood_diff > 0:
        combo_count += 1
        base_damage_reward = blood_diff * 8  # 提高基礎傷害獎勵
        combo_multiplier = min(combo_count * 0.3, 3.0)  # 提高連擊獎勵上限
        damage_reward = base_damage_reward * combo_multiplier
    else:
        combo_count = max(0, combo_count - 1)
        if action in [4, 5, 6]:  # 如果進行攻擊動作但沒造成傷害
            damage_reward = -3  # 輕微懲罰

    # 受傷懲罰
    damage_taken_reward = 0
    if next_self_blood - self_blood < -7:
        if stop == 0:
            damage_taken_reward = -3 * (self_blood - next_self_blood)
            safe_count = 0
            stop = 1
    else:
        stop = 0

    # 安全獎勵（降低）
    safe_reward = 0
    if next_self_blood == self_blood:
        safe_count += 1
        if safe_count >= 15:  # 增加安全計數門檻
            safe_reward = 5  # 降低安全獎勵
            safe_count = 0

    # 最終獎勵計算
    reward = (action_reward + 
             skill_reward + 
             damage_reward * 1.5 +  # 提高傷害獎勵權重
             damage_taken_reward + 
             safe_reward)
    
    # 回合衰減
    reward *= 0.995 ** target_step  # 降低衰減速度

    done = 0
    emergence_break = 0
    return reward, done, stop, emergence_break, safe_count, combo_count
def get_screen(screen_area):
    """ 擊取遊戲畫面 """
    screenshot = ImageGrab.grab(bbox=screen_area)
    frame = np.array(screenshot)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame

# Load a pretrained YOLO model
def detect_boss(frame, q):
    model = YOLO("best.pt")
    results = model.predict(source=frame, conf=0.7)

    # 預設 boss_position 為 None，表示未偵測到 BOSS
    boss_position = None

    for result in results:
        if result.boxes.xyxy.shape[0] > 0:  # 確認有檢測框
            boxes = result.boxes.xyxy  # 獲取預測框的坐標（xyxy: x1, y1, x2, y2）
            boss_position = [
                boxes[0, 0].item(),
                boxes[0, 1].item(),
                boxes[0, 2].item(),
                boxes[0, 3].item(),
                0
            ]  # 轉換為float
            break  # 假設只需要第一個框即可

    # 如果未偵測到，返回 None
    q.put(boss_position)

def adjust_view(boss_position, block, big_move):
    """ 調整視角使 BOSS 居中 """
    if boss_position is not None:
        x1, y1, x2, y2, _ = boss_position

        # 計算 BOSS 中心點相對於屏幕中心的位置
        x_center = (x1 + x2) / 2
        screen_center_x = frame_width // 2
        
        # 計算 BOSS 在屏幕左側還是右側
        is_boss_on_right = x_center > screen_center_x
        
        # 計算需要移動的距離
        distance = abs(x_center - screen_center_x)
        
        # 設置移動參數
        dead_zone = 50  # 增大死區
        max_move = 50   # 降低最大移動速度
        
        # 只有超出死區時才移動
        if distance > dead_zone:
            # 計算移動量，使用非線性縮放使移動更平滑
            move_multiplier = 2  # 移動倍數
            move_amount = min(distance - dead_zone, max_move) * move_multiplier
            
            # 根據 BOSS 位置決定移動方向並轉換為整數
            final_move = int(move_amount if is_boss_on_right else -move_amount)
            
            # 輸出調試信息
            print(f"Boss center: {x_center}")
            print(f"Screen center: {screen_center_x}")
            print(f"Is boss on right: {is_boss_on_right}")
            print(f"Move amount: {final_move}")
            
            # 執行移動
            win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, 
                               final_move, 0, 0, 0)
        block = 0
        big_move = 0
            
    else:
        block += 1
        if block >= (5 + 4 * big_move):
            win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, 
                                700, 0, 0, 0)
            big_move += 1
    return block, big_move

# PPO Parameters
PPO_model_path = "model_ppo"
PPO_log_path = "logs_ppo/"
WIDTH = 84
HEIGHT = 84
window_size = (435, 175, 1400, 800)  # screen region
boss_blood_window = (835, 215, 1085, 216)
self_blood_window = (881, 895, 1038, 896)
action_size = 7

EPISODES = 3000
UPDATE_STEP = 50
paused = True

# 螢幕中心點
screen_width = win32api.GetSystemMetrics(0)
screen_height = win32api.GetSystemMetrics(1)
center_x = screen_width // 2
center_y = screen_height // 2

# 遊戲畫面的解析度，根據你的顯示器解析度來設置
frame_width = 1920  # 遊戲畫面寬度
frame_height = 1080  # 遊戲畫面高度
# 用於螢幕擊取的範圍 (\u5168\u87a2\u5e55)
screen_area = (0, 0, frame_width, frame_height)

if __name__ == '__main__':
    # Ensure reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    agent = PPO(WIDTH, HEIGHT, action_size, PPO_model_path, PPO_log_path)

    # 初始化為暫停狀態
    paused = True
    print("Press J to start game...")
    while paused:  # 等待開始
        paused = pause_game(paused)
        time.sleep(0.1)

    emergence_break = 0
    detector = BackgroundObjectDetector()

    for episode in range(EPISODES):
        print(f'=======Episode: {episode} ========')
        screen_gray = cv2.cvtColor(grab_screen(window_size), cv2.COLOR_BGR2GRAY)
        boss_screen_gray = cv2.cvtColor(grab_screen(boss_blood_window), cv2.COLOR_BGR2GRAY)
        self_screen_gray = cv2.cvtColor(grab_screen(self_blood_window), cv2.COLOR_BGR2GRAY)
        station = cv2.resize(screen_gray, (WIDTH, HEIGHT))
        boss_blood = boss_blood_count(boss_screen_gray)
        self_blood = self_blood_count(self_screen_gray)

        last_time = time.time()

        done = 0
        total_reward = 0
        stop = 0
        target_step = 0
        num_step = 0
        safe_count = 0
        combo_count = 0
        e_in_cd, q_in_cd = False, False
        e_start_time, q_start_time = None, None
        block = 0
        big_move = 0

        while True:
            frame = get_screen(screen_area)
            detector.add_frame(frame)
            boss_position = detector.get_latest_result()
            block, big_move = adjust_view(boss_position, block, big_move)
            station = np.array(station).reshape(HEIGHT, WIDTH)  # Remove the extra dimension
            print('loop took {} seconds'.format(time.time() - last_time))
            last_time = time.time()
            target_step += 1
            action, log_prob = agent.choose_action(station)
            take_action(action)

            if action == 5:  # E
                if not e_in_cd:
                    e_start_time = time.time()
                    e_in_cd = True
                else:
                    current_time = time.time()
                    if current_time - e_start_time >= 10:
                        e_in_cd = False
            if action == 6:  # Q
                if not q_in_cd:
                    q_start_time = time.time()
                    q_in_cd = True
                else:
                    current_time = time.time()
                    if current_time - q_start_time >= 20:
                        q_in_cd = False

            # Take next state
            screen_gray = cv2.cvtColor(grab_screen(window_size), cv2.COLOR_BGR2GRAY)
            boss_blood_window_gray = cv2.cvtColor(grab_screen(boss_blood_window), cv2.COLOR_BGR2GRAY)
            self_blood_window_gray = cv2.cvtColor(grab_screen(self_blood_window), cv2.COLOR_BGR2GRAY)
            next_station = cv2.resize(screen_gray, (WIDTH, HEIGHT))
            next_station = np.array(next_station).reshape(HEIGHT, WIDTH)  # Ensure 2D shape
            next_station = torch.from_numpy(next_station).float()
            next_station = next_station.unsqueeze(0).unsqueeze(0)
            next_boss_blood = boss_blood_count(boss_blood_window_gray)
            next_self_blood = self_blood_count(self_blood_window_gray)

            reward, done, stop, emergence_break, safe_count, combo_count = action_judge(
                boss_blood, next_boss_blood, self_blood, next_self_blood, stop, emergence_break, e_in_cd, q_in_cd,
                safe_count, action, combo_count)

            if emergence_break == 100:
                print("Emergency break")
                agent.save_model()
                paused = True

            agent.store_data(station, action, reward, next_station, done, log_prob)
            if target_step % UPDATE_STEP == 0:
                agent.train()

            station = next_station
            self_blood = next_self_blood
            boss_blood = next_boss_blood
            total_reward += reward
            paused = pause_game(paused)

            if done == 1:
                break

        if episode % 10 == 0:
            agent.save_model()
        print('Episode: ', episode, 'Total Reward:', total_reward)
        restart()