import torch
import random
import numpy as np
import copy
from game import SnakeGameAI, Direction, Point
from model import LinearQNet
from helper import plot

# Parâmetros do Algoritmo Genético
POPULATION_SIZE = 100
TOP_K = 20 # Melhores para reprodução
SURVIVORS = 20 # Quantos sobrevivem (Elimina os 80 piores: 100 - 80 = 20)
MUTATION_RATE = 0.05
MUTATION_POWER = 0.2

class EvolutionAgent:
    def __init__(self, model=None):
        self.n_games = 0
        self.model = LinearQNet(12, 512, 3)
        if model:
            self.model.load_state_dict(model.state_dict())
        
    def get_state(self, game):
        # Mesma lógica de estado do DQN
        head = game.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)
        
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            (dir_r and game.is_collision(point_r)) or 
            (dir_l and game.is_collision(point_l)) or 
            (dir_u and game.is_collision(point_u)) or 
            (dir_d and game.is_collision(point_d)),

            (dir_u and game.is_collision(point_r)) or 
            (dir_d and game.is_collision(point_l)) or 
            (dir_l and game.is_collision(point_u)) or 
            (dir_r and game.is_collision(point_d)),

            (dir_d and game.is_collision(point_r)) or 
            (dir_u and game.is_collision(point_l)) or 
            (dir_r and game.is_collision(point_u)) or 
            (dir_l and game.is_collision(point_d)),
            
            dir_l, dir_r, dir_u, dir_d,
            
            game.food.x < game.head.x, 
            game.food.x > game.head.x, 
            game.food.y < game.head.y, 
            game.food.y > game.head.y,
            
            game.energy / 100.0
        ]
        return np.array(state, dtype=int)

    def get_action(self, state):
        # Ação determinística baseada na rede
        state0 = torch.tensor(state, dtype=torch.float)
        prediction = self.model(state0)
        move = torch.argmax(prediction).item()
        final_move = [0,0,0]
        final_move[move] = 1
        return final_move

def run_game(agent, ui_active=False):
    game = SnakeGameAI(render=ui_active)
    total_reward = 0
    while True:
        state = agent.get_state(game)
        action = agent.get_action(state)
        
        # O reward retornado pelo play_step agora considera as penalidades de energia
        reward, done, score = game.play_step(action, ui_active=ui_active)
        total_reward += reward
        
        if done:
            return total_reward, game.score # Retorna fitness (reward acumulado) e score visual

def train_genetic():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    generation = 0
    
    # 1. População Inicial
    population = []
    
    # Tenta carregar modelo existente para iniciar a população
    base_model = LinearQNet(12, 512, 3)
    loaded = base_model.load('best_model.pth')
    
    print("Inicializando população...")
    for i in range(POPULATION_SIZE):
        agent = EvolutionAgent()
        if loaded:
            # Se carregou, cria mutações do melhor modelo
            agent.model.load_state_dict(base_model.state_dict())
            if i > 0: # Mantém 1 cópia exata (elite), o resto muta
                agent.model.mutate(MUTATION_RATE, MUTATION_POWER)
        else:
            # Se não, começa aleatório
            pass 
        population.append(agent)
        
    while True:
        generation += 1
        scores = []
        
        # 2. Avaliação (Todos jogam)
        for i, agent in enumerate(population):
            # Só mostra a tela para a 1ª cobra da população (para ser rápido)
            show_ui = (i == 0)
            fitness, game_score = run_game(agent, ui_active=show_ui)
            # Armazena ((Fitness, ScoreReal), Agente)
            scores.append(((fitness, game_score), agent))
            
        # 3. Seleção
        # Ordena pelo FITNESS (total_reward) - Maior é melhor
        scores.sort(key=lambda x: x[0][0], reverse=True)
        
        # Estratégia: 
        # - Elimina os 5 piores (Mantém Top 15)
        # - Cria 5 novos a partir dos Top 5 (Breeding)
        
        survivors = scores[:SURVIVORS] # Top 15
        top_breeders = scores[:TOP_K]  # Top 5
        
        # Estatísticas da Geração
        best_fitness = scores[0][0][0]
        best_score_real = scores[0][0][1]
        
        avg_fitness = sum(s[0][0] for s in scores) / POPULATION_SIZE
        
        if best_score_real > record:
            record = best_score_real
            # Salva o melhor modelo (Agente com maior fitness costuma ter bom score, mas garantimos recorde real)
            # Idealmente salvaríamos o agente que fez o recorde. 
            # Mas aqui salvamos o Top Fitness. (Geralmente correlacionado)
            scores[0][1].model.save('best_model.pth')
            
        print(f"Geração {generation} | Fit: {best_fitness:.2f} | Score: {best_score_real} | Média Fit: {avg_fitness:.2f} | Recorde: {record}")
        
        plot_scores.append(best_score_real)
        total_score += best_score_real
        plot_mean_scores.append(total_score / generation)
        plot(plot_scores, plot_mean_scores)
        
        # 4. Reprodução
        new_population = []
        
        # Sobreviventes: Os Top 15 continuam vivos
        for item in survivors:
            agent = item[1]
            new_population.append(agent)
            
        # Filhos: Gera novos até completar a população (Faltam 5)
        while len(new_population) < POPULATION_SIZE:
            # Escolhe pai aleatório entre os Top 5
            parent_item = random.choice(top_breeders)
            parent_agent = parent_item[1]
            
            # Cria filho
            child_agent = EvolutionAgent(model=parent_agent.model)
            child_agent.model.mutate(MUTATION_RATE, MUTATION_POWER)
            
            new_population.append(child_agent)
            
        population = new_population

if __name__ == '__main__':
    train_genetic()
