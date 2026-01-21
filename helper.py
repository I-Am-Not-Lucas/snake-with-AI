import matplotlib.pyplot as plt

# Ativa modo interativo
plt.ion()

# Variáveis globais para manter a referência da figura
_fig = None
_ax = None
_line1 = None
_line2 = None

def plot(scores, mean_scores):
    global _fig, _ax, _line1, _line2
    
    # Se a figura ainda não existe (primeira vez), cria
    if _fig is None or not plt.fignum_exists(_fig.number):
        _fig, _ax = plt.subplots(figsize=(5, 4))
        _line1, = _ax.plot([], [], label='Score')
        _line2, = _ax.plot([], [], label='Média')
        _ax.set_xlabel('Gereção')
        _ax.set_ylabel('Score')
        _ax.legend()
        plt.show(block=False)

    # Atualiza apenas os dados (muito mais leve que recriar o gráfico)
    _line1.set_data(range(len(scores)), scores)
    _line2.set_data(range(len(mean_scores)), mean_scores)
    
    # Ajusta os limites da visualização
    _ax.relim()
    _ax.autoscale_view()
    
    # Atualiza o título com informações
    last_score = scores[-1] if scores else 0
    last_mean = mean_scores[-1] if mean_scores else 0
    _ax.set_title(f'Treinamento Genético - Atual: {last_score} | Média: {last_mean:.2f}')
    
    # Desenha as atualizações e processa eventos da interface
    _fig.canvas.draw()
    _fig.canvas.flush_events()
