import random
from collections import deque, namedtuple

import torch
from random_board import randomBoard
from encode_chat import encode_board, encode_legel_moves
import chess
from net import ChessDQN
from tqdm import tqdm

GEMMA = 0.9
Experience = namedtuple('Experience', ['state', 'action', 'next_state', 'reward'])

class ReplayStack:
  def __init__(self, capacity):
    self.buffer = deque(maxlen=capacity)

  def push(self, x):
    self.buffer.append(x)

  def sample(self, batch_size):
    if len(self.buffer) < batch_size:
      return self.buffer
    return random.sample(self.buffer, batch_size)


if __name__ == '__main__':
  board, moves = randomBoard(chess.Board(), 0, 100)
  target_board, more_moves = randomBoard(board, 1, 3)
  print('for board with ', moves, ' moves')
  print(board)
  print('more ', more_moves, ' moves, gives')
  print(target_board)
  print()
  
  net = ChessDQN()
  optimizer = torch.optim.AdamW(net.parameters(), lr=1e-3, amsgrad=True)
  replay_stack = ReplayStack(100)

  for _ in tqdm(range(100)):
    for _ in range(5):
      current_encoded = encode_board(board)
      target_encoded = encode_board(target_board)
      prediction = net(current_encoded.unsqueeze(0), target_encoded.unsqueeze(0))
      legal_mask = encode_legel_moves(board)
      prediction *= legal_mask

      if random.random() < 0.1:
        legal_indexes = torch.nonzero(legal_mask, as_tuple=False)
        chosen_move = random.choice(legal_indexes)
      else:
        argmax = torch.argmax(prediction)
        chosen_move = torch.tensor([argmax // 64, argmax % 64])
      move = chess.Move(*chosen_move)
      board.push(move)
      target_reached = board.board_fen() == target_board.board_fen()
      is_over = board.is_game_over()
      reward = 1 if target_reached else -1 if is_over else 0
      next_state = encode_board(board)
      experience = Experience(current_encoded, chosen_move, next_state, reward)
      replay_stack.push(experience)

      # Without replay buffer
      pred_q = prediction[0, *chosen_move]
      if target_reached:
        next_q = 1
      else:
        with torch.no_grad():
          next_q = reward + GEMMA * net(next_state.unsqueeze(0), target_encoded.unsqueeze(0)).max()
      criterion = torch.nn.SmoothL1Loss()
      loss = criterion(pred_q, next_q)

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

  test = chess.Board()
  test.push_uci('e2e4')
  test_target = chess.Board(test.fen())
  test_target.push_uci('e7e5')
  import IPython; IPython.embed()
