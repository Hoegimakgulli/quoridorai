import pygame
import PlayerType
import random

class PlayerData():
    def __init__(self, num, pos):
        self.player_num = num
        self.player_pos = pos

    def SetPlayerFrame(self, color, attackRange, moveRange, hp):
        self.player_color = color
        self.player_attackRange = attackRange
        self.player_moveRange = moveRange
        self.player_hp = hp

playerCount = 1
playerDatas = []
playerFrames = PlayerType.PlayerCreateFrame()

def PlayerSpawn():
    playerDatas.append(PlayerData(0, [4, 7]))
    frame = playerFrames[random.randint(0,6)]
    playerDatas[0].SetPlayerFrame(frame.player_color, \
        frame.player_attackRange, frame.player_moveRange, frame.player_hp)

def DrawPlayer(screen):
    for count in range(len(playerDatas)):
        drawPos = [playerDatas[count].player_pos[0] * 50 + 25, playerDatas[count].player_pos[1] * 50 + 25]
        pygame.draw.circle(screen, playerDatas[count].player_color, drawPos, 25)