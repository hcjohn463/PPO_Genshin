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


def pause_game(paused):
    keys = key_check()
    if 'T' in keys:
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
            if 'T' in keys:
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
    if action == 0:  # n_choose
        pass
    elif action == 1:  # w
        directkeys.go_forward()
    elif action == 2:  # s
        directkeys.go_back()
    elif action == 3:  # a
        directkeys.go_left()
    elif action == 4:  # d
        directkeys.go_right()
    elif action == 5:  # dodge
        directkeys.mouse_click('right')
    elif action == 6:  # normal attack
        directkeys.mouse_click('left')
    elif action == 7:  # elemental skill
        directkeys.skill()
    elif action == 8:  # elemental burst
        directkeys.burst()


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
    if action == 0:  # 不動
        action_reward = -5  # 懲罰不行動
    elif action in [1, 2, 3, 4]:  # 移動
        action_reward = -2  # 輕微懲罰過多移動
    elif action == 6:  # 普通攻擊
        action_reward = 5  # 鼓勵進行攻擊
    elif action == 5:  # 閃避
        blood_diff = self_blood - next_self_blood
        if blood_diff > 0:  # 如果閃避成功（沒受傷）
            action_reward = 10
        else:
            action_reward = -2

    # 技能獎勵機制
    skill_reward = 0
    if action == 7:  # E技能
        if e_in_cd:
            skill_reward = -20  # 嚴重懲罰在CD時使用E
        else:
            blood_diff = boss_blood - next_boss_blood
            if blood_diff > 0:
                skill_reward = 20  # 提高E技能命中獎勵
            else:
                skill_reward = -5
    
    elif action == 8:  # Q技能
        if q_in_cd:
            skill_reward = -30
        else:
            blood_diff = boss_blood - next_boss_blood
            if blood_diff > 0:
                skill_reward = 30  # 提高Q技能命中獎勵
            else:
                skill_reward = -10

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
        if action in [6, 7, 8]:  # 如果進行攻擊動作但沒造成傷害
            damage_reward = -3  # 輕微懲罰

    # 距離獎勵（簡化）
    position_reward = 0
    if distance_to_boss is not None:
        if distance_to_boss < 30:  # 縮小最小距離
            position_reward = -10
        elif distance_to_boss > 150:  # 縮小最大距離
            position_reward = -10
        else:
            position_reward = 2  # 降低正確距離的獎勵

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
             position_reward * 0.5 +  # 降低位置獎勵權重
             damage_taken_reward + 
             safe_reward)
    
    # 回合衰減
    reward *= 0.995 ** target_step  # 降低衰減速度

    done = 0
    emergence_break = 0
    return reward, done, stop, emergence_break, safe_count, combo_count


# PPO Parameters
PPO_model_path = "model_ppo"
PPO_log_path = "logs_ppo/"
WIDTH = 160
HEIGHT = 160
window_size = (435, 175, 1400, 800)  # screen region
boss_blood_window = (835, 215, 1085, 216)
self_blood_window = (881, 895, 1038, 896)
action_size = 9

EPISODES = 3000
UPDATE_STEP = 50
paused = True

if __name__ == '__main__':
    # Ensure reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    agent = PPO(WIDTH, HEIGHT, action_size, PPO_model_path, PPO_log_path)

    # 初始化為暫停狀態
    paused = True
    print("請按 T 開始遊戲...")
    while paused:  # 等待開始
        paused = pause_game(paused)
        time.sleep(0.1)

    emergence_break = 0

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
        safe_count = 0
        combo_count = 0
        e_in_cd, q_in_cd = False, False
        e_start_time, q_start_time = None, None

        while True:
            station = np.array(station).reshape(HEIGHT, WIDTH)  # Remove the extra dimension
            print('loop took {} seconds'.format(time.time() - last_time))
            last_time = time.time()
            target_step += 1
            action, log_prob = agent.choose_action(station)
            take_action(action)

            if action == 7:  # E
                if not e_in_cd:
                    e_start_time = time.time()
                    e_in_cd = True
                else:
                    current_time = time.time()
                    if current_time - e_start_time >= 5:
                        e_in_cd = False
            if action == 8:  # Q
                if not q_in_cd:
                    q_start_time = time.time()
                    q_in_cd = True
                else:
                    current_time = time.time()
                    if current_time - q_start_time >= 90:
                        q_in_cd = False

            # Take next state
            screen_gray = cv2.cvtColor(grab_screen(window_size), cv2.COLOR_BGR2GRAY)
            boss_blood_window_gray = cv2.cvtColor(grab_screen(boss_blood_window), cv2.COLOR_BGR2GRAY)
            self_blood_window_gray = cv2.cvtColor(grab_screen(self_blood_window), cv2.COLOR_BGR2GRAY)
            next_station = cv2.resize(screen_gray, (WIDTH, HEIGHT))
            next_station = np.array(next_station).reshape(HEIGHT, WIDTH)  # Ensure 2D shape
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
