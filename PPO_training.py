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
                 damage_count):
    if next_self_blood < 1:  # self dead
        if emergence_break < 2:
            reward = -20
            done = 1
            stop = 0
            emergence_break += 1
            return reward, done, stop, emergence_break, damage_count
        else:
            reward = -20
            done = 1
            stop = 0
            emergence_break = 100
            return reward, done, stop, emergence_break, damage_count
    else:
        self_blood_reward = 0
        boss_blood_reward = 0
        e_reward = -10 if e_in_cd else 0
        q_reward = -10 if q_in_cd else 0
        damage_reward = 0

        if next_self_blood == self_blood:
            damage_count += 1
            if damage_count == 15:
                damage_reward = 6
                damage_count = 0

        if next_self_blood - self_blood < -7:
            if stop == 0:
                self_blood_reward = -1
                damage_count = 0
                stop = 1
        else:
            stop = 0
        if next_boss_blood - boss_blood <= -3:
            boss_blood_reward = 10

        reward = self_blood_reward + boss_blood_reward + e_reward + q_reward + damage_reward
        done = 0
        emergence_break = 0
        return reward, done, stop, emergence_break, damage_count


# PPO Parameters
PPO_model_path = "model_ppo"
PPO_log_path = "logs_ppo/"
WIDTH = 96
HEIGHT = 88
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
    paused = pause_game(paused)
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
        damage_count = 0
        e_in_cd, q_in_cd = False, False
        e_start_time, q_start_time = None, None

        while True:
            station = np.array(station).reshape(HEIGHT, WIDTH)  # Remove the extra dimension
            print('loop took {} seconds'.format(time.time() - last_time))
            last_time = time.time()
            target_step += 1
            action, log_prob = agent.choose_action(station)
            take_action(action)
            print(f'Take action: {action}')

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

            reward, done, stop, emergence_break, damage_count = action_judge(
                boss_blood, next_boss_blood, self_blood, next_self_blood, stop, emergence_break, e_in_cd, q_in_cd,
                damage_count)

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
