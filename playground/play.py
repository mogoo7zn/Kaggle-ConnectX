import sys
import os
import pygame
import numpy as np
import time
import math

# Add project root to path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import agent and utils
# Allow selection of agent via environment variable or try multiple options
AGENT_MODULE = os.getenv('AGENT_MODULE', 'main')  # Can be 'main', 'main_alphazero', 'main_DQN', etc.

print(f"Loading AI agent from submission.{AGENT_MODULE}...")
print("(Note: main.py is very large and may take time to load)")
print("(Set AGENT_MODULE environment variable to use a different agent, e.g., 'main_alphazero')")

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
        # Try dynamic import
        module_name = f"submission.{AGENT_MODULE}"
        ai_agent = __import__(module_name, fromlist=[''])
    
    # Verify agent function exists
    if not hasattr(ai_agent, 'agent'):
        raise AttributeError(f"Module {AGENT_MODULE} does not have an 'agent' function")
    
    print(f"✓ AI agent loaded successfully from {AGENT_MODULE}")
    agent_loaded = True
    
except ImportError as e:
    print(f"✗ Error importing AI agent from {AGENT_MODULE}: {e}")
    print("\nTrying fallback options...")
    
    # Try fallback options
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
        except Exception as fallback_error:
            print(f"  ✗ {fallback} failed: {fallback_error}")
            continue
    
    if not agent_loaded:
        print("\n✗ Could not load any AI agent module.")
        print("Make sure you are running this from the playground directory or project root.")
        import traceback
        traceback.print_exc()
        sys.exit(1)
        
except AttributeError as e:
    print(f"✗ Agent module loaded but missing 'agent' function: {e}")
    print(f"Available attributes: {[attr for attr in dir(ai_agent) if not attr.startswith('_')]}")
    sys.exit(1)
    
except Exception as e:
    print(f"✗ Error loading AI agent: {e}")
    print("The agent file may be corrupted or incompatible.")
    import traceback
    traceback.print_exc()
    sys.exit(1)

try:
    from agents.base.config import config
    from agents.base import utils
    print("✓ Game utilities loaded successfully")
except ImportError as e:
    print(f"✗ Error importing game utilities: {e}")
    import traceback
    traceback.print_exc()
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
        except AttributeError as e:
            print(f"AI Error: agent function not found - {e}")
            print("Make sure submission/main.py has an 'agent' function defined.")
            import traceback
            traceback.print_exc()
            col = 0 # Fallback
        except Exception as e:
            print(f"AI Error: {e}")
            import traceback
            traceback.print_exc()
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

