import pygame
import player_character_type
import random

# 프로퍼티는 _"변수명"으로 설정
class PlayerData():
    def __init__(self, num, pos):
        self.player_num = num
        self.player_pos = pos

    def set_player_frame(self, color, attackRange, moveRange, player_hp):
        self.player_color = color
        self.player_attack_range = attackRange
        self.player_move_range = moveRange
        self._hp = player_hp

    @property
    def player_hp(self):
        return self._hp
    @player_hp.setter
    def player_hp(self, player_hp):
        global player_datas

        if self._hp <= 0:
            print("NUM : {} / player가 처치되었습니다.".format(self.player_num))
            for player in player_datas:
                if player.player_num == self.player_num:
                    player_datas.remove(player)
       
        self._hp = player_hp

player_count = 1
player_datas = []
player_frames = player_character_type.player_create_frame()

def player_spawn():
    player_datas.append(PlayerData(0, [4, 7]))
    frame = player_frames[random.randint(0,6)]
    player_datas[0].set_player_frame(frame.player_color, \
        frame.player_attack_range, frame.player_move_range, frame.player_hp)

def draw_player(screen):
    for count in range(len(player_datas)):
        draw_pos = [player_datas[count].player_pos[0] * 50 + 25, player_datas[count].player_pos[1] * 50 + 25]
        pygame.draw.circle(screen, player_datas[count].player_color, draw_pos, 25)