# Projeto de TCC: Redes Neurais com Snake Game

Este projeto implementa uma Inteligência Artificial capaz de aprender a jogar o clássico jogo da cobrinha (Snake) utilizando Aprendizado por Reforço (Deep Q-Learning).

## 📋 Funcionalidades
- **Rede Neural (Deep Q-Network)**: Aprende estratégias para maximizar a pontuação.
- **Sistema de Energia**: A cobra gasta energia ao se mover e recupera ao comer. Isso incentiva a IA a ser eficiente.
- **Dashboard em Tempo Real**:
  - Score Atual e Recorde (Melhor Cobra).
  - Barra de Energia Visual.
  - Gráfico de evolução do aprendizado.

## 🛠️ Tecnologias Utilizadas
- **Python 3.10+**
- **PyTorch**: Construção e treinamento da Rede Neural.
- **Pygame**: Interface gráfica e ambiente do jogo.
- **NumPy**: Manipulação matemática.
- **Matplotlib**: Plotagem de gráficos de performance.

## 🚀 Como Executar

1. **Instale as dependências**:
   Abra o terminal na pasta do projeto e execute:
   ```bash
   pip install -r requirements.txt
   ```

2. **Execute o treinamento**:
   ```bash
   python agent.py
   ```

A janela do jogo abrirá e você verá a IA aprendendo (errando muito no início e melhorando com o tempo). Um gráfico também será exibido mostrando a evolução da pontuação média.

## 📂 Estrutura dos Arquivos
- `agent.py`: O cérebro da IA. Contém o loop de treinamento e a tomada de decisão.
- `game.py`: O ambiente do jogo. Lógica da cobra, colisões e renderização.
- `model.py`: A arquitetura da Rede Neural (Linear Q-Net).
- `helper.py`: Utilitários para visualização de dados (gráficos).

---
**Nota**: O código está comentado detalhadamente para fins didáticos.
