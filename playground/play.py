import sys
import os
import pygame
import numpy as np
import time
import math

# Add project root to path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import agent and utils
try:
    import submission.main as ai_agent
    from agents.base.config import config
    from agents.base import utils
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Make sure you are running this from the playground directory or project root.")
    sys.exit(1)

# Colors
BLUE = (0, 0, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
YELLOW = (255, 255, 0)
WHITE = (255, 255, 255)
GRAY = (128, 128, 128)
LIGHT_BLUE = (173, 216, 230)

# Game Constants
SQUARESIZE = 100
RADIUS = int(SQUARESIZE/2 - 5)
width = config.COLUMNS * SQUARESIZE
height = (config.ROWS + 1) * SQUARESIZE
size = (width, height)

# Mock classes for Kaggle agent interface
class Observation:
    def __init__(self, board, mark):
        self.board = board
        self.mark = mark

class Configuration:
    def __init__(self, columns, rows, inarow, timeout=2):
        self.columns = columns
        self.rows = rows
        self.inarow = inarow
        self.timeout = timeout

class Button:
    def __init__(self, x, y, w, h, text, color, hover_color):
        self.rect = pygame.Rect(x, y, w, h)
        self.text = text
        self.color = color
        self.hover_color = hover_color
        self.font = pygame.font.SysFont("monospace", 30)

    def draw(self, screen):
        mouse_pos = pygame.mouse.get_pos()
        if self.rect.collidepoint(mouse_pos):
            pygame.draw.rect(screen, self.hover_color, self.rect)
        else:
            pygame.draw.rect(screen, self.color, self.rect)
        
        text_surf = self.font.render(self.text, True, BLACK)
        text_rect = text_surf.get_rect(center=self.rect.center)
        screen.blit(text_surf, text_rect)

    def is_clicked(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.rect.collidepoint(event.pos):
                return True
        return False

class Connect4UI:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode(size)
        pygame.display.set_caption("Connect4 vs AI")
        self.font = pygame.font.SysFont("monospace", 75)
        self.small_font = pygame.font.SysFont("monospace", 30)
        
        # Game State
        self.board = [0] * (config.ROWS * config.COLUMNS)
        self.game_over = False
        self.turn = 0 # 0 for Player (Red), 1 for AI (Yellow)
        self.winner = None
        
        # Buttons
        self.restart_btn = Button(width//2 - 100, height//2 + 50, 200, 50, "Restart", WHITE, GRAY)
        
    def draw_board(self):
        for c in range(config.COLUMNS):
            for r in range(config.ROWS):
                pygame.draw.rect(self.screen, BLUE, (c*SQUARESIZE, r*SQUARESIZE + SQUARESIZE, SQUARESIZE, SQUARESIZE))
                pygame.draw.circle(self.screen, BLACK, (int(c*SQUARESIZE + SQUARESIZE/2), int(r*SQUARESIZE + SQUARESIZE + SQUARESIZE/2)), RADIUS)
        
        board_2d = np.array(self.board).reshape(config.ROWS, config.COLUMNS)
        for c in range(config.COLUMNS):
            for r in range(config.ROWS):
                # In pygame grid, row 0 is top, but visual board connects from bottom.
                # Actually standard visual is row 0 top.
                # Let's match the visual to the board array.
                # Array: row 0 is top.
                # Pygame coords: y=0 is top.
                # So piece at board[r][c] goes to y = (r+1)*SQUARESIZE
                
                piece = board_2d[r][c]
                if piece == 1:
                    pygame.draw.circle(self.screen, RED, (int(c*SQUARESIZE + SQUARESIZE/2), int(r*SQUARESIZE + SQUARESIZE + SQUARESIZE/2)), RADIUS)
                elif piece == 2: 
                    pygame.draw.circle(self.screen, YELLOW, (int(c*SQUARESIZE + SQUARESIZE/2), int(r*SQUARESIZE + SQUARESIZE + SQUARESIZE/2)), RADIUS)
        
        pygame.display.update()

    def run(self):
        self.draw_board()
        
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    sys.exit()
                
                if event.type == pygame.MOUSEMOTION:
                    if not self.game_over and self.turn == 0:
                        pygame.draw.rect(self.screen, BLACK, (0, 0, width, SQUARESIZE))
                        posx = event.pos[0]
                        pygame.draw.circle(self.screen, RED, (posx, int(SQUARESIZE/2)), RADIUS)
                    pygame.display.update()
                
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if not self.game_over and self.turn == 0:
                        pygame.draw.rect(self.screen, BLACK, (0, 0, width, SQUARESIZE))
                        posx = event.pos[0]
                        col = int(math.floor(posx/SQUARESIZE))
                        
                        if utils.is_valid_move(self.board, col):
                            # Player Drop Piece
                            self.board = utils.make_move(self.board, col, 1)
                            
                            if utils.check_winner(self.board, 1):
                                self.game_over = True
                                self.winner = 1
                                self.draw_board()
                                self.show_message("Player Wins!", RED)
                            
                            self.turn = 1
                            self.draw_board()
                            
                            if not self.game_over:
                                self.handle_ai_turn()

                    if self.game_over:
                        if self.restart_btn.is_clicked(event):
                            self.reset_game()

    def handle_ai_turn(self):
        # Show thinking
        pygame.draw.rect(self.screen, BLACK, (0, 0, width, SQUARESIZE))
        label = self.small_font.render("AI Thinking...", 1, YELLOW)
        self.screen.blit(label, (10, 10))
        pygame.display.update()
        
        # AI Move
        obs = Observation(self.board, 2)
        conf = Configuration(config.COLUMNS, config.ROWS, config.INAROW)
        
        try:
            start_time = time.time()
            col = ai_agent.agent(obs, conf)
            end_time = time.time()
            print(f"AI chose column {col} in {end_time - start_time:.4f}s")
        except Exception as e:
            print(f"AI Error: {e}")
            col = 0 # Fallback
            
        if utils.is_valid_move(self.board, col):
            self.board = utils.make_move(self.board, col, 2)
            
            if utils.check_winner(self.board, 2):
                self.game_over = True
                self.winner = 2
                self.draw_board()
                self.show_message("AI Wins!", YELLOW)
            
            self.turn = 0
            self.draw_board()
        else:
            print(f"AI attempted invalid move: {col}")
            # Fallback or end game?
            # Let's just skip turn or pick first valid
            valid = utils.get_valid_moves(self.board)
            if valid:
                self.board = utils.make_move(self.board, valid[0], 2)
                self.turn = 0
                self.draw_board()
    
    def show_message(self, text, color):
        pygame.draw.rect(self.screen, BLACK, (0, 0, width, SQUARESIZE))
        label = self.font.render(text, 1, color)
        rect = label.get_rect(center=(width/2, SQUARESIZE/2))
        self.screen.blit(label, rect)
        
        # Show restart button
        self.restart_btn.draw(self.screen)
        
        pygame.display.update()
        
    def reset_game(self):
        self.board = [0] * (config.ROWS * config.COLUMNS)
        self.game_over = False
        self.turn = 0
        self.winner = None
        self.draw_board()

if __name__ == "__main__":
    game = Connect4UI()
    game.run()

