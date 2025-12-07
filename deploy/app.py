import sys
import os
import pygame
import numpy as np
import time
import math
import importlib

# --- Path Setup for Standalone/Dev ---
if getattr(sys, 'frozen', False):
    # If the application is run as a bundle, the PyInstaller bootloader
    # extends the sys module by a flag frozen=True and sets the app 
    # path into variable _MEIPASS'.
    base_path = sys._MEIPASS
else:
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if base_path not in sys.path:
    sys.path.append(base_path)

# Try to import config and utils
try:
    from agents.base.config import config
    from agents.base import utils
except ImportError as e:
    # Fallback config if imports fail (e.g. strict packaging issues)
    class Config:
        ROWS = 6
        COLUMNS = 7
        INAROW = 4
    config = Config()
    
    # Minimal utils fallback
    class Utils:
        def get_valid_moves(self, board):
            return [c for c in range(config.COLUMNS) if board[c] == 0]
        def is_valid_move(self, board, col):
            return board[col] == 0
        def make_move(self, board, col, mark):
            new_board = list(board)
            for r in range(config.ROWS-1, -1, -1):
                if new_board[r * config.COLUMNS + col] == 0:
                    new_board[r * config.COLUMNS + col] = mark
                    break
            return new_board
        def check_winner(self, board, mark):
            # Simplified check (horizontal, vertical, diagonal)
            b = np.array(board).reshape(config.ROWS, config.COLUMNS)
            # Horizontal
            for r in range(config.ROWS):
                for c in range(config.COLUMNS - 3):
                    if np.all(b[r, c:c+4] == mark): return True
            # Vertical
            for r in range(config.ROWS - 3):
                for c in range(config.COLUMNS):
                    if np.all(b[r:r+4, c] == mark): return True
            # Diag
            for r in range(config.ROWS - 3):
                for c in range(config.COLUMNS - 3):
                    if np.all([b[r+i, c+i] == mark for i in range(4)]): return True
                    if np.all([b[r+3-i, c+i] == mark for i in range(4)]): return True
            return False
            
    utils = Utils()

# --- Constants & Colors ---
# Modern Dark Theme
BG_COLOR_TOP = (44, 62, 80)    # Midnight Blue
BG_COLOR_BOT = (30, 33, 40)    # Darker
BOARD_COLOR = (52, 73, 94)     # Wet Asphalt
BOARD_SHADOW = (44, 62, 80)
SLOT_COLOR = (20, 23, 30)      # Dark slot
P1_COLOR = (231, 76, 60)       # Alizarin Red
P1_SHADOW = (192, 57, 43)
P2_COLOR = (241, 196, 15)      # Sun Flower Yellow
P2_SHADOW = (243, 156, 18)
TEXT_COLOR = (236, 240, 241)   # Clouds
HIGHLIGHT_COLOR = (46, 204, 113) # Emerald Green (for winning line)

SQUARESIZE = 100
RADIUS = int(SQUARESIZE/2 - 8)
WIDTH = config.COLUMNS * SQUARESIZE
HEIGHT = (config.ROWS + 1) * SQUARESIZE
SIZE = (WIDTH, HEIGHT)

# --- Helper Classes ---

class Button:
    def __init__(self, x, y, w, h, text, color=(52, 152, 219), hover_color=(41, 128, 185), text_color=(255, 255, 255), font_size=24):
        self.rect = pygame.Rect(x, y, w, h)
        self.text = text
        self.base_color = color
        self.hover_color = hover_color
        self.text_color = text_color
        self.font = pygame.font.SysFont("Segoe UI", font_size, bold=True)
        self.is_hovered = False

    def draw(self, screen):
        mouse_pos = pygame.mouse.get_pos()
        self.is_hovered = self.rect.collidepoint(mouse_pos)
        
        color = self.hover_color if self.is_hovered else self.base_color
        
        # Draw shadow
        shadow_rect = self.rect.copy()
        shadow_rect.y += 4
        pygame.draw.rect(screen, (0, 0, 0, 100), shadow_rect, border_radius=12)
        
        # Draw button
        pygame.draw.rect(screen, color, self.rect, border_radius=12)
        
        # Draw text
        text_surf = self.font.render(self.text, True, self.text_color)
        text_rect = text_surf.get_rect(center=self.rect.center)
        screen.blit(text_surf, text_rect)

    def is_clicked(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                return True
        return False

class DropAnimation:
    def __init__(self, col, row, color, shadow_color):
        self.col = col
        self.target_row = row
        self.color = color
        self.shadow_color = shadow_color
        self.x = int(col * SQUARESIZE + SQUARESIZE/2)
        self.y = int(SQUARESIZE/2) # Start from top
        self.target_y = int(row * SQUARESIZE + SQUARESIZE + SQUARESIZE/2)
        self.velocity = 0
        self.gravity = 2
        self.finished = False
        self.bounce = False

    def update(self):
        self.velocity += self.gravity
        self.y += self.velocity
        
        if self.y >= self.target_y:
            if not self.bounce:
                self.y = self.target_y
                self.velocity = -self.velocity * 0.4 # Bounce
                self.bounce = True
                if abs(self.velocity) < 5:
                    self.finished = True
            else:
                self.y = self.target_y
                self.finished = True

    def draw(self, screen):
        # Draw shadow/3D effect
        pygame.draw.circle(screen, self.shadow_color, (self.x, int(self.y)+3), RADIUS)
        pygame.draw.circle(screen, self.color, (self.x, int(self.y)), RADIUS)
        # Add shine
        pygame.draw.circle(screen, (255, 255, 255, 100), (self.x - RADIUS//3, int(self.y) - RADIUS//3), RADIUS//4)

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

class Connect4UI:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode(SIZE)
        pygame.display.set_caption("Connect X - Ultimate")
        
        # Fonts
        self.title_font = pygame.font.SysFont("Segoe UI", 60, bold=True)
        self.status_font = pygame.font.SysFont("Segoe UI", 36, bold=True)
        self.small_font = pygame.font.SysFont("Segoe UI", 20)
        
        # Game State
        self.state = "MENU" 
        self.board = [0] * (config.ROWS * config.COLUMNS)
        self.turn = 0 
        self.winner = None
        self.human_player_idx = 0 
        self.mouse_x = WIDTH // 2
        self.animations = []
        
        # Agent Selection
        self.available_agents = {
            "AlphaZero (Ultra)": "main_alphazero",
            "DQN Agent": "main_DQN",
            "Minimax (Simple)": "main_backup"
        }
        self.selected_agent_name = "AlphaZero (Ultra)"
        self.ai_agent_module = None
        
        # Initialize UI Elements
        self.init_menu()
        self.init_game_ui()

    def init_menu(self):
        cx = WIDTH // 2
        cy = HEIGHT // 2
        
        self.btn_play_human_first = Button(cx - 150, cy - 60, 300, 50, "Play as Red (First)", color=P1_COLOR, hover_color=(255, 100, 80))
        self.btn_play_ai_first = Button(cx - 150, cy + 10, 300, 50, "Play as Yellow (Second)", color=P2_COLOR, hover_color=(255, 220, 50), text_color=(50, 50, 50))
        
        self.btn_agent_toggle = Button(cx - 250, cy + 100, 500, 40, f"Opponent: {self.selected_agent_name}", color=(100, 100, 100))

    def init_game_ui(self):
        self.btn_menu = Button(20, HEIGHT - 80, 120, 40, "Menu", color=(149, 165, 166))
        self.btn_restart = Button(WIDTH - 140, HEIGHT - 80, 120, 40, "Restart", color=(46, 204, 113))

    def load_agent(self):
        module_name = self.available_agents[self.selected_agent_name]
        print(f"Loading agent: {module_name}...")
        try:
            # Dynamic import
            if module_name == 'main':
                import submission.main as agent
            elif module_name == 'main_alphazero':
                import submission.main_alphazero as agent
            elif module_name == 'main_DQN':
                import submission.main_DQN as agent
            elif module_name == 'main_backup':
                import submission.main_backup as agent
            else:
                agent = importlib.import_module(f"submission.{module_name}")
            
            self.ai_agent_module = agent
            print("Agent loaded.")
            # Reset button style on success
            self.btn_agent_toggle.base_color = (100, 100, 100)
            self.btn_agent_toggle.text = f"Opponent: {self.selected_agent_name}"
            return True
        except Exception as e:
            print(f"Failed to load agent {module_name}: {e}")
            self.btn_agent_toggle.text = "Load Error! (See Console)"
            self.btn_agent_toggle.base_color = (200, 50, 50)
            return False

    def reset_game(self):
        self.board = [0] * (config.ROWS * config.COLUMNS)
        self.winner = None
        self.turn = 0 
        self.mouse_x = WIDTH // 2
        self.animations = []
        
        # Ensure agent is loaded
        if not self.ai_agent_module:
            if not self.load_agent():
                # Stay in menu if load fails
                return False
        return True

    def get_row_for_col(self, col):
        # Find the first empty row from bottom
        for r in range(config.ROWS-1, -1, -1):
            if self.board[r * config.COLUMNS + col] == 0:
                return r
        return -1

    def draw_gradient_bg(self):
        # Simple vertical gradient
        for y in range(HEIGHT):
            alpha = y / HEIGHT
            r = int(BG_COLOR_TOP[0] * (1-alpha) + BG_COLOR_BOT[0] * alpha)
            g = int(BG_COLOR_TOP[1] * (1-alpha) + BG_COLOR_BOT[1] * alpha)
            b = int(BG_COLOR_TOP[2] * (1-alpha) + BG_COLOR_BOT[2] * alpha)
            pygame.draw.line(self.screen, (r,g,b), (0, y), (WIDTH, y))

    def draw_board(self):
        self.draw_gradient_bg()
        
        # Draw Board Container
        board_rect = pygame.Rect(0, SQUARESIZE, WIDTH, HEIGHT-SQUARESIZE)
        pygame.draw.rect(self.screen, BOARD_COLOR, board_rect)
        
        # Draw Slots
        for c in range(config.COLUMNS):
            for r in range(config.ROWS):
                cx = int(c*SQUARESIZE + SQUARESIZE/2)
                cy = int(r*SQUARESIZE + SQUARESIZE + SQUARESIZE/2)
                
                idx = r * config.COLUMNS + c
                piece = self.board[idx]
                
                # Draw hole background (dark)
                pygame.draw.circle(self.screen, SLOT_COLOR, (cx, cy), RADIUS)
                
                if piece != 0:
                    color = P1_COLOR if piece == 1 else P2_COLOR
                    shadow = P1_SHADOW if piece == 1 else P2_SHADOW
                    
                    # Check if this is the target of an active animation
                    is_animating = False
                    for anim in self.animations:
                        if anim.col == c and anim.target_row == r and not anim.finished:
                            is_animating = True
                            break
                    
                    if not is_animating:
                        # Draw Shadow
                        pygame.draw.circle(self.screen, shadow, (cx, cy+3), RADIUS)
                        # Draw Piece
                        pygame.draw.circle(self.screen, color, (cx, cy), RADIUS)
                        # Shine
                        pygame.draw.circle(self.screen, (255, 255, 255, 80), (cx - RADIUS//3, cy - RADIUS//3), RADIUS//4)

        # Draw Animations
        for anim in self.animations:
            anim.draw(self.screen)

        # Draw Top Bar
        pygame.draw.rect(self.screen, (0,0,0,50), (0, 0, WIDTH, SQUARESIZE))
        
        if self.state == "PLAYING":
            if self.turn == self.human_player_idx:
                # Floating piece
                color = P1_COLOR if self.human_player_idx == 0 else P2_COLOR
                pygame.draw.circle(self.screen, color, (self.mouse_x, int(SQUARESIZE/2)), RADIUS)
            else:
                text = f"AI ({self.selected_agent_name}) Thinking..."
                label = self.status_font.render(text, True, TEXT_COLOR)
                self.screen.blit(label, (20, 25))
                
        elif self.state == "GAME_OVER":
            # Overlay
            s = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
            s.fill((0,0,0,150))
            self.screen.blit(s, (0,0))
            
            if self.winner == 1:
                text = "Red Wins!"
                color = P1_COLOR
            elif self.winner == 2:
                text = "Yellow Wins!"
                color = P2_COLOR
            else:
                text = "Draw!"
                color = TEXT_COLOR
                
            label = self.title_font.render(text, True, color)
            label_rect = label.get_rect(center=(WIDTH//2, HEIGHT//2 - 50))
            self.screen.blit(label, label_rect)
            
            self.btn_menu.rect.centerx = WIDTH//2 - 80
            self.btn_menu.rect.centery = HEIGHT//2 + 50
            self.btn_restart.rect.centerx = WIDTH//2 + 80
            self.btn_restart.rect.centery = HEIGHT//2 + 50
            
            self.btn_menu.draw(self.screen)
            self.btn_restart.draw(self.screen)

        pygame.display.update()

    def handle_ai_turn(self):
        # Force redraw
        self.draw_board()
        pygame.event.pump()
        
        ai_mark = 2 if self.human_player_idx == 0 else 1
        
        obs = Observation(self.board, ai_mark)
        conf = Configuration(config.COLUMNS, config.ROWS, config.INAROW)
        
        col = 0
        try:
            if self.ai_agent_module:
                col = self.ai_agent_module.agent(obs, conf)
            else:
                # Fallback random
                valid = utils.get_valid_moves(self.board)
                col = valid[0] if valid else 0
        except Exception as e:
            print(f"AI Error: {e}")
            valid = utils.get_valid_moves(self.board)
            col = valid[0] if valid else 0

        if utils.is_valid_move(self.board, col):
            row = self.get_row_for_col(col)
            
            # Start animation
            color = P1_COLOR if ai_mark == 1 else P2_COLOR
            shadow = P1_SHADOW if ai_mark == 1 else P2_SHADOW
            self.animations.append(DropAnimation(col, row, color, shadow))
            
            # Update board immediately logic-wise, but visual is handled by animation check
            self.board = utils.make_move(self.board, col, ai_mark)
            
            if utils.check_winner(self.board, ai_mark):
                self.winner = ai_mark
                # Delay game over state until animation finishes?
                # We'll handle that in update loop
            
            self.turn = 1 - self.turn
        else:
            print(f"AI Invalid move {col}")

    def run(self):
        clock = pygame.time.Clock()
        
        while True:
            dt = clock.tick(60)
            
            # Update animations
            for anim in self.animations[:]:
                anim.update()
                if anim.finished:
                    # Keep it in list? No, remove it, and let the static board draw it.
                    # But we need to make sure the static board draws it now.
                    # Since we updated self.board already, it will be drawn.
                    self.animations.remove(anim)
                    
                    # Check game over condition delayed
                    if self.winner is not None and not self.animations:
                        self.state = "GAME_OVER"

            if self.state == "MENU":
                self.draw_gradient_bg()
                
                # Title
                title = self.title_font.render("Connect X", True, TEXT_COLOR)
                title_rect = title.get_rect(center=(WIDTH//2, HEIGHT//4))
                self.screen.blit(title, title_rect)
                
                self.btn_play_human_first.draw(self.screen)
                self.btn_play_ai_first.draw(self.screen)
                self.btn_agent_toggle.draw(self.screen)
                
                pygame.display.update()
                
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        sys.exit()
                    
                    if self.btn_play_human_first.is_clicked(event):
                        self.human_player_idx = 0
                        if self.reset_game():
                            self.state = "PLAYING"
                        
                    if self.btn_play_ai_first.is_clicked(event):
                        self.human_player_idx = 1
                        if self.reset_game():
                            self.state = "PLAYING"
                        
                    if self.btn_agent_toggle.is_clicked(event):
                        # Cycle agents
                        names = list(self.available_agents.keys())
                        curr_idx = names.index(self.selected_agent_name)
                        next_idx = (curr_idx + 1) % len(names)
                        self.selected_agent_name = names[next_idx]
                        self.btn_agent_toggle.text = f"Opponent: {self.selected_agent_name}"
                        # Reload agent if needed
                        self.ai_agent_module = None

            elif self.state == "PLAYING":
                # Check if animations are running, maybe block input?
                # Optional: Block input while animating
                is_animating = len(self.animations) > 0
                
                is_ai_turn = (self.turn != self.human_player_idx)
                
                if is_ai_turn and not is_animating and self.winner is None:
                    pygame.time.wait(300)
                    self.handle_ai_turn()
                
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        sys.exit()
                    
                    if not is_ai_turn and self.winner is None:
                        if event.type == pygame.MOUSEMOTION:
                            self.mouse_x = event.pos[0]
                        
                        if event.type == pygame.MOUSEBUTTONDOWN:
                            if is_animating: continue # Prevent spamming
                            
                            self.mouse_x = event.pos[0]
                            col = int(math.floor(self.mouse_x/SQUARESIZE))
                            
                            if 0 <= col < config.COLUMNS:
                                human_mark = 1 if self.human_player_idx == 0 else 2
                                
                                if utils.is_valid_move(self.board, col):
                                    row = self.get_row_for_col(col)
                                    
                                    # Animation
                                    color = P1_COLOR if human_mark == 1 else P2_COLOR
                                    shadow = P1_SHADOW if human_mark == 1 else P2_SHADOW
                                    self.animations.append(DropAnimation(col, row, color, shadow))
                                    
                                    self.board = utils.make_move(self.board, col, human_mark)
                                    
                                    if utils.check_winner(self.board, human_mark):
                                        self.winner = human_mark
                                    
                                    self.turn = 1 - self.turn
                
                self.draw_board()
            
            elif self.state == "GAME_OVER":
                self.draw_board() # Handles overlay
                
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        sys.exit()
                    
                    if self.btn_menu.is_clicked(event):
                        self.state = "MENU"
                    
                    if self.btn_restart.is_clicked(event):
                        if self.reset_game():
                            self.state = "PLAYING"

if __name__ == "__main__":
    game = Connect4UI()
    game.run()
