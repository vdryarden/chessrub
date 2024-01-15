
import random
import chess

def randomBoard(board, moves_start, moves_num):
	board = chess.Board(board.fen())
	moves_len = random.randint(moves_start, moves_num)
	for _ in range(moves_len):
		moves = list(board.legal_moves)
		if not moves:
			break

		board.push(random.choice(moves))

	return board, moves_len
