# import pgzrun
import ship as lander
import terrain as land
import utils

WIDTH = 1400
HEIGHT = 800
DIM = (WIDTH, HEIGHT)
PI = 3.1415926

Black = (0, 0, 0)
White = (255, 255, 255)

# Game class
class Game:
    # Initialize game state
    def __init__(self):
        # Variables for game state
        self.score = 0
        self.state = 1
        self.multiplier = 1
        self.gas = 0
        self.playing = True

        # Used to see if there is no collision (0), a crash (1), or a potential landing (2)
        self.collided = 0

        # Variables for physics
        self.dt = .01
        self.xVel = 0
        self.yVel = 0
        self.x = 0
        self.y = 0
        self.ang = 0

        # Used to see what kind of landing: good landing (1), hard landing (2), or a crash (3)
        self.landingType = 0

        # Ship and terrain objects
        self.ship = lander.Ship()
        self.terrain = land.Terrain()
        self.xTerrain = []
        self.yTerrain = []

        # Used to schedule a life reset after the landing/crash screen
        self.resetScheduled = False

    # Reset game state
    def resetGame(self):
        self.state = 2
        self.score = 0
        self.ship.setGas(750)
        self.resetLife()

    # Reset settings for new life
    def resetLife(self):
        self.ship.setPos(100, 100)
        self.ship.setVel(50, 0)
        self.ship.setAng(0)
        self.ship.setAccMode(0)
        self.terrain.generate(WIDTH, HEIGHT, 0, 1)
        self.collided = 0
        self.multiplier = 1
        self.xTerrain = self.terrain.getXPoints()
        self.yTerrain = self.terrain.getYPoints()
        self.playing = True
        self.resetScheduled = False

    def execute_action(self, action=0):
        # angle of our ship
        # Range (-PI, 0)
        def angleReward(value):
            return 0.0

        # velocity of the ship
        # value is a Tuple(a, b) here, watch out!
        # Range xVel (0, 100) yVel (-unknown, 103.333)
        def velocityReward(value):
            return 0.0

        # gas left
        # Range (0, 750)
        def gasReward(value):
            return 0.0

        # distance from ship to the center
        # Range (0, 700)
        def distanceReward(value):
            return 0.0

        state = self
        reward = 0
        isGameEnd = 0

        reward -= 1

        if action == 0:
            self.customUpdate(Game.Input())
        elif action == 1:
            self.customUpdate(Game.Input(left=1))
        elif action == 2:
            self.customUpdate(Game.Input(right=1))
        elif action == 3:
            self.customUpdate(Game.Input(up=1))
        elif action == 4:
            self.customUpdate(Game.Input(down=1))

        if self.gas > 0:
            # If the user has a good landing
            if self.landingType == 1:
                reward += 500
            # If the user has a hard landing
            elif self.landingType == 2:
                reward += 200
            # Too fast resulting in a crash
            elif self.landingType == 3:
                reward -= 200

            reward += angleReward(self.ang)
            reward += velocityReward((self.xVel, self.yVel))
            reward += gasReward(self.gas)
            reward += distanceReward(abs(self.ship.xpos - WIDTH / 2))
        else:
            reward -= 200

        isGameEnd = (self.state == 3)

        return self.get_ship_info(), reward, isGameEnd

    # Callback for switching from game over screen
    def gameOver(self):
        self.resetScheduled = False
        self.state = 1

    class GameInfo:
        def __init__(self, ship, terrain):
            self.ship = ship
            self.terrain = terrain

    def getGameInfo(self):
        return Game.GameInfo(self.ship, self.terrain)

    def get_ship_info(self):
        return self.ship.xpos, self.ship.ypos, self.ship.xVel, self.ship.yVel, self.ship.ang, self.ship.gas, self.ship.accMode


    class Input:
        def __init__(self, left=0, right=0, up=0, down=0):
            self.p = 0
            self.q = 0
            self.left = left
            self.right = right
            self.up = up
            self.down = down

    # Update game state manually
    def customUpdate(self, input: Input, dt = 1 / 60):
        self.dt = dt

        # Menu state. Wait for 'p' to start game
        if self.state == 1:
            if input.p:
                self.resetGame()

        # Game state. Update game variables, update ship, check collisions
        elif self.state == 2:
            if self.playing:
                # Get the fuel, velocities, positions, and angle
                self.gas = self.ship.getGas()
                self.xVel = self.ship.getXvel()
                self.yVel = self.ship.getYvel()
                self.x = self.ship.getXpos()
                self.y = self.ship.getYpos()
                self.ang = self.ship.getAng()

                self.y = self.y + self.yVel * self.dt + .5 * 30. * self.dt * self.dt
                self.x = self.x + self.xVel * self.dt

                # If ship goes off on the side of the screen, it is moved to the other side.
                if self.x > WIDTH + 10:
                    self.x = -10
                elif self.x < -10:
                    self.x = WIDTH + 10

                # If the ship goes too high, then it is placed back at the initial spawn point
                if self.y < -50:
                    self.x = 100
                    self.y = 100
                    self.xVel = 50
                    self.yVel = 0
                    self.ship.setAng(0)
                    self.ship.setAccMode(0)

                # Makes x velocity stay within - 100 and 100 inclusive
                if self.xVel >= 100:
                    self.xVel = 100
                elif self.xVel <= -100:
                    self.xVel = -100

                # Consider gravity
                self.yVel = self.yVel + 10. * self.dt

                # Calculate new velocities based on ship acceleration
                self.xVel, self.yVel = self.ship.accelerate(self.xVel, self.yVel)

                # Update ship's new velocities and position
                self.ship.setPos(self.x, self.y)
                self.ship.setVel(self.xVel, self.yVel)

                # Left Arrow. Rotate left.
                if input.left:
                    self.ship.rotate(self.ang - PI / 14.)
                # Right Arrow. Rotate right.
                elif input.right:
                    self.ship.rotate(self.ang + PI / 14.)
                # Down Arrow. Decrease rocket thrust
                elif input.down:
                    self.ship.accelerateChangeWithoutSound(-1)
                # Up arrow. Increase rocket thrust
                elif input.up:
                    self.ship.accelerateChangeWithoutSound(1)
                # Q key; Effectively quit program
                elif input.q:
                    self.state = 1

                # Recalculates ship's hitbox
                self.ship.hitbox()

                # Checks if the ship collided with the terrain
                self.collided = 0
                self.collided = self.ship.collision(self.xTerrain, self.yTerrain)

                # We just play once ever turn
                if self.collided != 0:
                    self.state = 3

                # Crashed
                if self.collided == 1:
                    self.playing = False
                    self.score += 5
                    self.ship.setGas(self.ship.getGas() - 100)
                    self.gas = self.ship.getGas()
                # Potential landing
                elif self.collided == 2:
                    self.playing = False

                    # Checks if the ship landed on a multiplier spot
                    self.multiplier = self.terrain.multiplierCheck(self.ship.getXpos())

                    # If the user has a good landing
                    if self.yVel < 12 and abs(self.xVel) < 25:
                        self.landingType = 1
                        self.ship.setGas(self.ship.getGas() + 50)
                        self.score += self.multiplier * 50
                    # If the user has a hard landing
                    elif self.yVel < 25 and abs(self.xVel) < 25:
                        self.landingType = 2
                        self.score += self.multiplier * 15
                    # If the user was going too fast resulting in a crash
                    else:
                        self.landingType = 3
                        self.score += 5
                        self.ship.setGas(self.ship.getGas() - 100)

                    # Update game state gas for display
                    self.gas = self.ship.getGas()

                # Out of gas
                if self.ship.getGas() <= 0:
                    self.gas = self.ship.getGas()
                    self.state = 3

    def customDraw(self, path):
        import pygame as pg

        pg.init()

        # create the display window
        win = pg.display.set_mode(DIM)
        win.fill(Black)

        pg.display.set_caption("Lunar Lander")

        self.ship.customDraw(win)
        self.terrain.customDraw(win)

        scoreStr = "SCORE    " + str(self.score)
        gasStr = "FUEL       " + str(int(self.gas))
        altStr = "ALTITUDE                      " + str(int(HEIGHT - 10 - self.y))
        xVelStr = "HORIZONTAL SPEED    " + str(int(self.xVel))
        yVelStr = "VERTICAL SPEED         " + str(int(-self.yVel))

        utils.drawText(win, scoreStr, (40, 40))
        utils.drawText(win, gasStr, (40, 60))
        utils.drawText(win, altStr, (WIDTH - 260, 40))
        utils.drawText(win, xVelStr, (WIDTH - 260, 60))
        utils.drawText(win, yVelStr, (WIDTH - 260, 80))


        # If the ship collided with the terrain, display the correct message
        if self.collided == 1:
            utils.drawText(
                win,
                "YOU CRASHED\nYOU LOST 100 FUEL UNITS",
                (WIDTH / 2 - 130, HEIGHT / 2 - 30))
            if not self.resetScheduled:
                self.resetScheduled = True
                self.resetLife()
        elif self.collided == 2:
            if self.landingType == 1:
                utils.drawText(
                    win,
                    "GOOD LANDING\n50 FUEL UNITS ADDED",
                    (WIDTH / 2 - 100, HEIGHT / 2 - 30))
            elif self.landingType == 2:
                utils.drawText(
                    win,
                    "HARD LANDING",
                    (WIDTH / 2 - 75, HEIGHT / 2 - 30))
            elif self.landingType == 3:
                utils.drawText(
                    win,
                    "YOU CRASHED\nYOU LOST 100 FUEL UNITS",
                    (WIDTH / 2 - 130, HEIGHT / 2 - 30))
            if not self.resetScheduled:
                self.resetScheduled = True
                self.resetLife()

        pg.image.save(win, path)

# Main logic for drawing
def draw():
    screen.clear()

    # Main menu state
    if game.state == 1:
        screen.draw.text("LUNAR LANDER", (WIDTH / 2 - 250, HEIGHT / 2 - 60), fontsize=80, fontname="dylova")
        screen.draw.text("Press P to play", (WIDTH / 2 - 70, HEIGHT / 2 + 50), fontname="dylova")
        screen.draw.text("Controls:\nLeft and Right Arrows: Rotate Ship\nUp and Down Arrows: Strengthen/Weaken Thrusters\nQ: Quit game",
                         (WIDTH / 2 - 270, HEIGHT / 2 + 120), align = "center",fontname="dylova")

    # Playing state
    elif game.state == 2:
        game.ship.draw(screen)
        game.terrain.draw(screen)
        scoreStr = "SCORE    " + str(game.score)
        gasStr = "FUEL       " + str(int(game.gas))
        altStr = "ALTITUDE                      " + str(int(HEIGHT - 10 - game.y))
        xVelStr = "HORIZONTAL SPEED    " + str(int(game.xVel))
        yVelStr = "VERTICAL SPEED         " + str(int(-game.yVel))
        screen.draw.text(scoreStr, (40, 40),fontsize=20, fontname="dylova")
        screen.draw.text(gasStr, (40, 60), fontsize=20, fontname="dylova")
        screen.draw.text(altStr, (WIDTH - 260, 40), fontsize=20, fontname="dylova")
        screen.draw.text(xVelStr, (WIDTH - 260, 60), fontsize=20, fontname="dylova")
        screen.draw.text(yVelStr, (WIDTH - 260, 80), fontsize=20, fontname="dylova")

        # If the ship collided with the terrain, display the correct message
        if game.collided == 1:
            screen.draw.text("YOU CRASHED\nYOU LOST 100 FUEL UNITS", (WIDTH / 2 - 130, HEIGHT / 2 - 30),
                             align="center", fontname="dylova")
            if not game.resetScheduled:
                game.resetScheduled = True
                clock.schedule(game.resetLife, 4.0)
        elif game.collided == 2:
            if game.landingType == 1:
                screen.draw.text("GOOD LANDING\n50 FUEL UNITS ADDED", (WIDTH / 2 - 100, HEIGHT / 2 - 30),
                                 align="center", fontname="dylova")
            elif game.landingType == 2:
                screen.draw.text("HARD LANDING", (WIDTH / 2 - 75, HEIGHT / 2 - 30), fontname="dylova")
            elif game.landingType == 3:
                screen.draw.text("YOU CRASHED\nYOU LOST 100 FUEL UNITS", (WIDTH / 2 - 130, HEIGHT / 2 - 30),
                                 align="center", fontname="dylova")
            if not game.resetScheduled:
                game.resetScheduled = True
                clock.schedule(game.resetLife, 4.0)

    # Game over state
    elif game.state == 3:
        game.ship.draw(screen)
        game.terrain.draw(screen)
        scoreStr = "SCORE    " + str(game.score)
        gasStr = "FUEL       " + str(int(game.gas))
        altStr = "ALTITUDE                      " + str(int(HEIGHT - 10 - game.y))
        xVelStr = "HORIZONTAL SPEED    " + str(int(game.xVel))
        yVelStr = "VERTICAL SPEED         " + str(int(-game.yVel))
        screen.draw.text(scoreStr, (40, 40),fontsize=20, fontname="dylova")
        screen.draw.text(gasStr, (40, 60), fontsize=20, fontname="dylova")
        screen.draw.text(altStr, (WIDTH - 260, 40), fontsize=20, fontname="dylova")
        screen.draw.text(xVelStr, (WIDTH - 260, 60), fontsize=20, fontname="dylova")
        screen.draw.text(yVelStr, (WIDTH - 260, 80), fontsize=20, fontname="dylova")

        screen.draw.text("YOU RAN OUT OF FUEL\nGAME OVER", (WIDTH / 2-115, HEIGHT / 2 - 30), align= "center", fontname="dylova")
        if not game.resetScheduled:
            game.resetScheduled = True
            clock.schedule(game.gameOver, 5.0)

# Update game state
def update(dt):
    game.dt = dt

    # Menu state. Wait for 'p' to start game
    if game.state == 1:
        if keyboard.p:
            game.resetGame()

    # Game state. Update game variables, update ship, check collisions
    elif game.state == 2:
        if game.playing:
            # Get the fuel, velocities, positions, and angle
            game.gas = game.ship.getGas()
            game.xVel = game.ship.getXvel()
            game.yVel = game.ship.getYvel()
            game.x = game.ship.getXpos()
            game.y = game.ship.getYpos()
            game.ang = game.ship.getAng()

            game.y = game.y + game.yVel * game.dt + .5 * 30. * game.dt * game.dt
            game.x = game.x + game.xVel * game.dt

            # If ship goes off on the side of the screen, it is moved to the other side.
            if game.x > WIDTH + 10:
                game.x = -10
            elif game.x < -10:
                game.x = WIDTH + 10

            # If the ship goes too high, then it is placed back at the initial spawn point
            if game.y < -50:
                game.x = 100
                game.y = 100
                game.xVel = 50
                game.yVel = 0
                game.ship.setAng(0)
                game.ship.setAccMode(0)

            # Makes x velocity stay within - 100 and 100 inclusive
            if game.xVel >= 100:
                game.xVel = 100
            elif game.xVel <= -100:
                game.xVel = -100

            # Consider gravity
            game.yVel = game.yVel + 10. * game.dt

            # Calculate new velocities based on ship acceleration
            game.xVel, game.yVel = game.ship.accelerate(game.xVel, game.yVel)

            # Update ship's new velocities and position
            game.ship.setPos(game.x, game.y)
            game.ship.setVel(game.xVel, game.yVel)

            # Left Arrow. Rotate left.
            if keyboard.left:
                game.ship.rotate(game.ang - PI/14.)
            # Right Arrow. Rotate right.
            elif keyboard.right:
                game.ship.rotate(game.ang + PI/14.)
            # Down Arrow. Decrease rocket thrust
            elif keyboard.down:
                game.ship.accelerateChange(-1, sounds)
            # Up arrow. Increase rocket thrust
            elif keyboard.up:
                game.ship.accelerateChange(1, sounds)
            # Q key; Effectively quit program
            elif keyboard.q:
                game.state = 1
                sounds.rocket_thrust.stop()

            # Recalculates ship's hitbox
            game.ship.hitbox()

            # Checks if the ship collided with the terrain
            game.collided = 0
            game.collided = game.ship.collision(game.xTerrain, game.yTerrain)

            # We just play once ever turn
            if game.collided != 0:
                game.state = 3

            # Crashed
            if game.collided == 1:
                sounds.rocket_thrust.stop()
                game.playing = False
                game.score += 5
                game.ship.setGas(game.ship.getGas() - 100)
                game.gas = game.ship.getGas()
                sounds.explosion.play()
                print(game.ship.xpos, game.ship.ypos)
            # Potential landing
            elif game.collided == 2:
                sounds.rocket_thrust.stop()
                game.playing = False

                # Checks if the ship landed on a multiplier spot
                game.multiplier = game.terrain.multiplierCheck(game.ship.getXpos())

                # If the user has a good landing
                if game.yVel < 12 and abs(game.xVel) < 25:
                    game.landingType = 1
                    game.ship.setGas(game.ship.getGas() + 50)
                    game.score += game.multiplier * 50
                # If the user has a hard landing
                elif game.yVel < 25 and abs(game.xVel) < 25:
                    game.landingType = 2
                    game.score += game.multiplier * 15
                # If the user was going too fast resulting in a crash
                else:
                    game.landingType = 3
                    game.score += 5
                    game.ship.setGas(game.ship.getGas() - 100)
                    sounds.explosion.play()

                # Update game state gas for display
                game.gas = game.ship.getGas()

            # Out of gas
            if game.ship.getGas() <= 0:
                sounds.rocket_thrust.stop()
                game.gas = game.ship.getGas()
                game.state = 3

# Run game
# pgzrun.go()

if __name__ == "__main__":

    game = Game()
    game.resetGame()

    # for i in range(500):
    #     game.customUpdate(Game.Input(up=1))

    while game.state != 3:
        game.customUpdate(Game.Input(up=1))

    print(game.yVel)

    game.customDraw("lunar_lander.png")



