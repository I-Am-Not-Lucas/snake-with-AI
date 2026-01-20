import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np
import math

pygame.init()
# Use SysFont to avoid requiring a local .ttf file
font = pygame.font.SysFont('arial', 25)

# Enum para direções
class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

# Ponto cartesiano
Point = namedtuple('Point', 'x, y')

# Cores (RGB)
WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
YELLOW = (255, 255, 0)

# Constantes do jogo
BLOCK_SIZE = 20
SPEED = 20
MAX_ENERGY = 200
ENERGY_STEP_COST = 1
ENERGY_FOOD_REWARD = 40
FOOD_REWARD = 20
COLLISION_PENALTY = -10
ENERGY_DEPLETION_PENALTY = -50


class SnakeGameAI:
    
    
    def __init__(self, w=640, h=480, render=True):
        self.w = w
        self.h = h
        # Init display only if render is True or if it doesn't exist yet
        self.display = None
        if render:
            self.display = pygame.display.set_mode((self.w, self.h))
            pygame.display.set_caption('Snake')
            
        self.clock = pygame.time.Clock()
        self.reset()
        self.best_score = 0
        
    def reset(self):
        """
        Reinicia o estado do jogo para o início de um novo episódio de treinamento.
        É chamado sempre que a cobra morre (Game Over).
        """
        # Define a direção inicial como Direita
        self.direction = Direction.RIGHT
        
        # Define a posição inicial da cabeça no centro da tela
        self.head = Point(self.w/2, self.h/2)
        
        # Cria o corpo inicial da cobra com 3 blocos
        self.snake = [self.head, 
                      Point(self.head.x-BLOCK_SIZE, self.head.y),
                      Point(self.head.x-(2*BLOCK_SIZE), self.head.y)]
        
        self.score = 0
        self.food = None
        self._place_food()
        
        # Sistema de Energia: Recarrega a energia máxima
        self.energy = MAX_ENERGY
        # Contador de frames para evitar loops infinitos
        self.frame_iteration = 0
        
        # Inicializa distância para reward shaping
        self.prev_distance = math.sqrt((self.head.x - self.food.x)**2 + (self.head.y - self.food.y)**2)
        
    def _place_food(self):
        # Coloca comida em posição aleatória
        x = random.randint(0, (self.w-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE 
        y = random.randint(0, (self.h-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()
        
    def play_step(self, action, ui_active=True):
        """
        Executa um passo no jogo dado uma ação da IA.
        Retorna: reward (recompensa), game_over (booleano), score (pontuação)
        """
        self.frame_iteration += 1
        
        # 1. Coleta eventos do usuário (Permite fechar a janela)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        
        # 2. Movimento: Atualiza a posição da cabeça baseada na ação
        self._move(action) 
        self.snake.insert(0, self.head) # Adiciona nova cabeça na lista
        
        # 3. Custo de Energia: Cada movimento custa energia vital
        self.energy -= ENERGY_STEP_COST
        
        reward = 0
        game_over = False
        
        # 4. Checa condições de derrota (Game Over)
        reward = 0
        game_over = False
        
        # Prioridade de Game Over:
        # 1. Energia acabou (Punição grave por inanição)
        if self.energy <= 0:
            game_over = True
            reward = ENERGY_DEPLETION_PENALTY 
        # 2. Colisão ou Timeout (Punição padrão)
        elif self.is_collision() or self.frame_iteration > 100*len(self.snake):
            game_over = True
            reward = COLLISION_PENALTY
            
        if game_over:
            return reward, game_over, self.score
            
        # 5. Lógica da Comida
        if self.head == self.food:
            self.score += 1
            if self.score > self.best_score:
                self.best_score = self.score
            reward = FOOD_REWARD # Recompensa por comer
            
            # Recupera energia ao comer, limitado ao máximo permitido
            self.energy = min(MAX_ENERGY, self.energy + ENERGY_FOOD_REWARD)
            
            self._place_food()
        else:
            # --- REWARD SHAPING (Modelagem de Recompensa) ---
            # Dá um feedback se a cobra está indo na direção certa ou errada
            
            # Distância anterior (antes do movimento atual)
            # Para calcular precisamos simular onde a cabeça estava. 
            # Mas como já movemos, podemos comparar a distância ATUAL com a distância se não tivéssemos movido?
            # Simplificação: Comparar com a distância do food em relação à cabeça ANTIGA.
            # Como não guardamos a cabeça antiga aqui facilmente sem mudar a assinatura,
            # vamos usar uma heurística simples baseada no estado atual.
            
            # Melhor abordagem sem refatorar tudo: Calcular dist a cada step e armazenar self.prev_distance
            pass 
            
            # Se não comeu, remove o último pedaço da cauda
            self.snake.pop()
        
        # --- CÁLCULO DE RECOMPENSA EXTRA ---
        # Recalcula distância atual
        head_x, head_y = self.head.x, self.head.y
        food_x, food_y = self.food.x, self.food.y
        current_distance = math.sqrt((head_x - food_x)**2 + (head_y - food_y)**2)
        
        if hasattr(self, 'prev_distance'):
            if current_distance < self.prev_distance:
                reward += 0.5 # Está chegando perto! (Aumentado para acelerar aprendizado)
            else:
                reward -= 0.5 # Está se afastando!
        
        self.prev_distance = current_distance
        
        # Punição leve por existência (incentiva velocidade e eficiência)
        reward -= 0.01
        
        # 6. Atualiza a tela e controla o FPS (SE A UI ESTIVER ATIVA)
        if ui_active:
            self._update_ui()
            self.clock.tick(SPEED)
        
        return reward, game_over, self.score
    
    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        # Bateu nas bordas?
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
        # Bateu em si mesma?
        if pt in self.snake[1:]:
            return True
        
        return False
        
    def _update_ui(self):
        if self.display is None:
            # Tenta recuperar o display se ele já existe (criado por outro agente)
            self.display = pygame.display.get_surface()
            if self.display is None:
                return # Sem display, sem UI

        self.display.fill(BLACK)
        
        # Desenha Cobra
        for pt in self.snake:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x+4, pt.y+4, 12, 12))
            
        # Desenha Comida
        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))
        
        # --- DASHBOARD ---
        # Score Atual
        text_score = font.render(f"Score: {self.score}", True, WHITE)
        self.display.blit(text_score, [0, 0])
        
        # Melhor Score (Recorde)
        text_best = font.render(f"Best: {self.best_score}", True, YELLOW)
        self.display.blit(text_best, [0, 30])
        
        # Energia
        # Cor varia conforme a energia: Verde (>50), Amarelo (>20), Vermelho (<20)
        energy_color = GREEN
        if self.energy < 50:
            energy_color = YELLOW
        if self.energy < 20:
            energy_color = RED
            
        text_energy = font.render(f"Energy: {self.energy}", True, energy_color)
        self.display.blit(text_energy, [self.w - 180, 0])
        
        # Barra de Energia
        pygame.draw.rect(self.display, WHITE, pygame.Rect(self.w - 180, 35, 150, 20), 2) # Borda
        # O comprimento da barra é proporcional à energia
        energy_bar_width = int((self.energy / MAX_ENERGY) * 146) 
        if energy_bar_width > 0:
            pygame.draw.rect(self.display, energy_color, pygame.Rect(self.w - 178, 37, energy_bar_width, 16))
        
        pygame.display.flip()
        
    def _move(self, action):
        # Action é [straight, right, left]
        
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)
        
        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx] # sem mudança
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx] # direita virar r -> d -> l -> u
        else: # [0, 0, 1]
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx] # esquerda virar r -> u -> l -> d
            
        self.direction = new_dir
        
        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE
            
        self.head = Point(x, y)
