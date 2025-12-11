import sys
import os
import pygame
import numpy as np
import time
import math

# Add project root to path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import agent and utils
AGENT_MODULE = os.getenv('AGENT_MODULE', 'main')

print(f"Loading AI agent from submission.{AGENT_MODULE}...")
print("(Note: main.py is very large and may take time to load)")

ai_agent = None
agent_loaded = False

# Try to load the specified agent
try:
    if AGENT_MODULE == 'main':
        import submission.main as ai_agent
    elif AGENT_MODULE == 'main_alphazero':
        import submission.main_alphazero as ai_agent
    elif AGENT_MODULE == 'main_DQN':
        import submission.main_DQN as ai_agent
    else:
        module_name = f"submission.{AGENT_MODULE}"
        ai_agent = __import__(module_name, fromlist=[''])
    
    if not hasattr(ai_agent, 'agent'):
        raise AttributeError(f"Module {AGENT_MODULE} does not have an 'agent' function")
    
    print(f"✓ AI agent loaded successfully from {AGENT_MODULE}")
    agent_loaded = True
    
except ImportError as e:
    print(f"✗ Error importing AI agent from {AGENT_MODULE}: {e}")
    print("\nTrying fallback options...")
    fallback_modules = ['main_alphazero', 'main_DQN', 'main_backup']
    for fallback in fallback_modules:
        try:
            print(f"  Trying {fallback}...")
            if fallback == 'main_alphazero':
                import submission.main_alphazero as ai_agent
            elif fallback == 'main_DQN':
                import submission.main_DQN as ai_agent
            elif fallback == 'main_backup':
                import submission.main_backup as ai_agent
            
            if hasattr(ai_agent, 'agent'):
                print(f"  ✓ Successfully loaded {fallback}")
                agent_loaded = True
                break
        except Exception:
            continue
    
    if not agent_loaded:
        print("\n✗ Could not load any AI agent module.")
        sys.exit(1)
        
except Exception as e:
    print(f"✗ Error loading AI agent: {e}")
    sys.exit(1)

try:
    from agents.base.config import config
    from agents.base import utils
    print("✓ Game utilities loaded successfully")
except ImportError as e:
    print(f"✗ Error importing game utilities: {e}")
    sys.exit(1)

# --- Modern Colors ---
BG_COLOR = (30, 33, 40)        # Dark Grey/Blue Background
BOARD_COLOR = (65, 105, 225)   # Royal Blue
SLOT_COLOR = (20, 23, 30)      # Darker slot color (empty)
P1_COLOR = (231, 76, 60)       # Flat Red
P2_COLOR = (241, 196, 15)      # Flat Yellow
TEXT_COLOR = (236, 240, 241)   # Off-white
BUTTON_COLOR = (52, 152, 219)  # Blue button
BUTTON_HOVER = (41, 128, 185)  # Darker blue hover
BUTTON_TEXT = (255, 255, 255)

# Game Constants
SQUARESIZE = 100
RADIUS = int(SQUARESIZE/2 - 8) # Slightly smaller for cleaner look
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
    def __init__(self, x, y, w, h, text, color=BUTTON_COLOR, hover_color=BUTTON_HOVER, text_color=BUTTON_TEXT, font_size=30):
        self.rect = pygame.Rect(x, y, w, h)
        self.text = text
        self.color = color
        self.hover_color = hover_color
        self.text_color = text_color
        self.font = pygame.font.SysFont("Segoe UI", font_size, bold=True)
        if not pygame.font.get_fonts(): # Fallback if Segoe UI not found
             self.font = pygame.font.SysFont("arial", font_size, bold=True)

    def draw(self, screen):
        mouse_pos = pygame.mouse.get_pos()
        current_color = self.hover_color if self.rect.collidepoint(mouse_pos) else self.color
        
        # Draw rounded rect (simulated by drawing rect + circles or just standard rect for simplicity)
        pygame.draw.rect(screen, current_color, self.rect, border_radius=10)
        
        # Text
        text_surf = self.font.render(self.text, True, self.text_color)
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
        pygame.display.set_caption("Connect X - AI Playground")
        
        # Fonts
        self.title_font = pygame.font.SysFont("Segoe UI", 60, bold=True)
        self.status_font = pygame.font.SysFont("Segoe UI", 40, bold=True)
        self.small_font = pygame.font.SysFont("Segoe UI", 24)
        
        # Game State
        self.state = "MENU" # MENU, PLAYING, GAME_OVER
        self.board = [0] * (config.ROWS * config.COLUMNS)
        self.turn = 0 # 0 for Player 1, 1 for Player 2
        self.winner = None
        self.human_player_idx = 0 # 0 if human is P1, 1 if human is P2
        self.mouse_x = width // 2 # Track mouse position for floating piece
        
        # Menu Buttons
        btn_width = 300
        btn_height = 60
        center_x = width // 2 - btn_width // 2
        
        self.btn_human_first = Button(center_x, height//2 - 80, btn_width, btn_height, "Human First (Red)")
        self.btn_ai_first = Button(center_x, height//2 + 20, btn_width, btn_height, "AI First (Yellow)")
        
        # Game Over Buttons
        self.btn_menu = Button(width//2 - 160, height//2 + 20, 150, 50, "Menu", color=(100, 100, 100), hover_color=(120, 120, 120))
        self.btn_restart = Button(width//2 + 10, height//2 + 20, 150, 50, "Play Again")

    def reset_game(self):
        self.board = [0] * (config.ROWS * config.COLUMNS)
        self.winner = None
        self.turn = 0 # Always start with P1 turn logic, but who P1 is depends on selection
        self.mouse_x = width // 2
        
    def draw_menu(self):
        self.screen.fill(BG_COLOR)
        
        # Title
        title = self.title_font.render("Connect X", True, TEXT_COLOR)
        title_rect = title.get_rect(center=(width//2, height//4))
        self.screen.blit(title, title_rect)
        
        # Subtitle
        sub = self.small_font.render("Select Game Mode", True, (180, 180, 180))
        sub_rect = sub.get_rect(center=(width//2, height//4 + 60))
        self.screen.blit(sub, sub_rect)
        
        self.btn_human_first.draw(self.screen)
        self.btn_ai_first.draw(self.screen)
        
        pygame.display.update()

    def draw_board(self):
        self.screen.fill(BG_COLOR)
        
        # Draw Board Background
        board_rect = pygame.Rect(0, SQUARESIZE, width, height-SQUARESIZE)
        
        # Draw slots
        for c in range(config.COLUMNS):
            for r in range(config.ROWS):
                # Calculate center
                cx = int(c*SQUARESIZE + SQUARESIZE/2)
                cy = int(r*SQUARESIZE + SQUARESIZE + SQUARESIZE/2)
                
                # Draw blue square container
                pygame.draw.rect(self.screen, BOARD_COLOR, (c*SQUARESIZE, r*SQUARESIZE + SQUARESIZE, SQUARESIZE, SQUARESIZE))
                
                # Draw circle (empty or filled)
                # Map 1D board to 2D coordinates
                # Board index: r * COLUMNS + c
                idx = r * config.COLUMNS + c
                piece = self.board[idx]
                
                color = SLOT_COLOR
                if piece == 1:
                    color = P1_COLOR
                elif piece == 2:
                    color = P2_COLOR
                
                pygame.draw.circle(self.screen, color, (cx, cy), RADIUS)

        # Draw Top Bar (Status)
        pygame.draw.rect(self.screen, BG_COLOR, (0, 0, width, SQUARESIZE))
        
        if self.state == "PLAYING":
            if self.turn == self.human_player_idx:
                # Draw floating piece for human
                color = P1_COLOR if self.human_player_idx == 0 else P2_COLOR
                pygame.draw.circle(self.screen, color, (self.mouse_x, int(SQUARESIZE/2)), RADIUS)
            else:
                text = "AI Thinking..."
                color = P2_COLOR if self.human_player_idx == 0 else P1_COLOR
                label = self.status_font.render(text, True, color)
                self.screen.blit(label, (20, 25))
            
        elif self.state == "GAME_OVER":
            if self.winner == 1:
                text = "Player 1 Wins!"
                color = P1_COLOR
            elif self.winner == 2:
                text = "Player 2 Wins!"
                color = P2_COLOR
            else:
                text = "Draw!"
                color = TEXT_COLOR
                
            # Center the result text
            label = self.status_font.render(text, True, color)
            label_rect = label.get_rect(center=(width//2, SQUARESIZE//2))
            self.screen.blit(label, label_rect)
            
            # Draw buttons
            self.btn_menu.draw(self.screen)
            self.btn_restart.draw(self.screen)

        pygame.display.update()

    def handle_ai_turn(self):
        # Force a redraw to show "AI Thinking"
        self.draw_board()
        pygame.event.pump() # Process event queue to prevent freezing
        
        # AI is always the player that is NOT the human
        ai_mark = 2 if self.human_player_idx == 0 else 1
        
        obs = Observation(self.board, ai_mark)
        conf = Configuration(config.COLUMNS, config.ROWS, config.INAROW)
        
        try:
            start_time = time.time()
            col = ai_agent.agent(obs, conf)
            end_time = time.time()
            print(f"AI (P{ai_mark}) chose column {col} in {end_time - start_time:.4f}s")
        except Exception as e:
            print(f"AI Error: {e}")
            col = 0
            # Try to find first valid move
            valid = utils.get_valid_moves(self.board)
            if valid: col = valid[0]

        if utils.is_valid_move(self.board, col):
            self.board = utils.make_move(self.board, col, ai_mark)
            
            if utils.check_winner(self.board, ai_mark):
                self.state = "GAME_OVER"
                self.winner = ai_mark
            
            # Switch turn
            self.turn = 1 - self.turn # Toggle 0/1
        else:
            print(f"AI attempted invalid move: {col}")
            # Fallback
            valid = utils.get_valid_moves(self.board)
            if valid:
                self.board = utils.make_move(self.board, valid[0], ai_mark)
                self.turn = 1 - self.turn

    def run(self):
        clock = pygame.time.Clock()
        
        while True:
            clock.tick(60) # Limit FPS
            
            if self.state == "MENU":
                self.draw_menu()
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        sys.exit()
                    
                    if self.btn_human_first.is_clicked(event):
                        self.human_player_idx = 0 # Human is P1 (Red)
                        self.reset_game()
                        self.state = "PLAYING"
                        
                    if self.btn_ai_first.is_clicked(event):
                        self.human_player_idx = 1 # Human is P2 (Yellow)
                        self.reset_game()
                        self.state = "PLAYING"
            
            elif self.state == "PLAYING":
                # Check if it's AI's turn
                is_ai_turn = (self.turn != self.human_player_idx)
                
                if is_ai_turn:
                    # Add a small delay so it doesn't feel instant/glitchy
                    pygame.time.wait(500)
                    self.handle_ai_turn()
                else:
                    # Human Turn
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            sys.exit()
                        
                        if event.type == pygame.MOUSEMOTION:
                            self.mouse_x = event.pos[0]
                        
                        if event.type == pygame.MOUSEBUTTONDOWN:
                            self.mouse_x = event.pos[0]
                            col = int(math.floor(self.mouse_x/SQUARESIZE))
                            
                            human_mark = 1 if self.human_player_idx == 0 else 2
                            
                            if utils.is_valid_move(self.board, col):
                                self.board = utils.make_move(self.board, col, human_mark)
                                
                                if utils.check_winner(self.board, human_mark):
                                    self.state = "GAME_OVER"
                                    self.winner = human_mark
                                
                                self.turn = 1 - self.turn
                    
                    self.draw_board()
            
            elif self.state == "GAME_OVER":
                self.draw_board() # Draw final state with overlay
                
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        sys.exit()
                    
                    if self.btn_menu.is_clicked(event):
                        self.state = "MENU"
                    
                    if self.btn_restart.is_clicked(event):
                        self.reset_game()
                        self.state = "PLAYING"

if __name__ == "__main__":
    game = Connect4UI()
    game.run()
