# -*- coding: utf-8 -*-
"""
Created on Tue Jul 17 14:07:02 2018

@author: Administrator
"""
import pygame
pygame.init()
screen=pygame.display.set_caption('hello world')
screen=pygame.display.set_mode([1200,800])
screen.fill([255,255,255])
pygame.draw.circle(screen,[0,0,0],[100,50],7,0) #颜色、坐标、半径、是否填充
points=[]
for i in range(5):
    for j in range(6):
        point=[(i+1)*200,(j+1)*100]
        points.append(point)
        pygame.draw.circle(screen,[0,0,0],point,7,0)
pygame.draw.line(screen,[0,0,0],[100,50],[200,100],3)
for i in range(5):
    pygame.draw.line(screen,[0,0,0],[200*i+200,100],[200*i+200,600],3)
for i in range(6):
    pygame.draw.line(screen,[0,0,0],[200,100*i+100],[1000,100*i+100],3)

pygame.draw.circle(screen,[255,0,0],[600,200],240,1) 
pygame.draw.circle(screen,[255,0,0],[800,400],200,1) 
pygame.draw.circle(screen,[255,0,0],[400,500],300,1)
pygame.display.flip()
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            
            
            

