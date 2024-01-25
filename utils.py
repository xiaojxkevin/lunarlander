import lunarLander


def drawText(win, text, pos):
    import pygame as pg
    font = pg.font.Font('fonts/dylova.ttf', 20)
    text = font.render(text, True, lunarLander.White, lunarLander.Black)
    win.blit(text, pos)
