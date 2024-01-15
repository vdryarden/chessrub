import torch
import chess

def encode_board(chess_board):
    """
    Convert a chessboard represented as a python-chess Board to a one-hot encoded PyTorch tensor.

    Parameters:
    - chess_board (chess.Board): The chess board represented using the python-chess library.

    Returns:
    - tensor (torch.Tensor): One-hot encoded PyTorch tensor representing the chessboard.
    """
    # Map the python-chess piece values to custom integer identifiers
    piece_mapping = {
        chess.PAWN: 0,
        chess.KNIGHT: 1,
        chess.BISHOP: 2,
        chess.ROOK: 3,
        chess.QUEEN: 4,
        chess.KING: 5,
    }

    # Create an empty 6D array to represent the one-hot encoded chessboard
    one_hot_chessboard = torch.zeros((12, 8, 8), dtype=torch.float32)

    # Populate the one-hot encoded chessboard array based on the python-chess Board
    for square in chess.SQUARES:
        piece = chess_board.piece_at(square)
        if piece is not None:
            piece_id = piece_mapping.get(piece.piece_type, 6)  # Use 6 for an unknown piece type
            color_channel = 0 if piece.color == chess.WHITE else 6  # Use channels 0-5 for white pieces, 6-11 for black pieces
            one_hot_chessboard[color_channel + piece_id, 7 - chess.square_rank(square), chess.square_file(square)] = 1

    return one_hot_chessboard


def encode_legel_moves(board: chess.Board):
    result = torch.zeros(64, 64)
    for move in board.legal_moves:
        result[move.from_square, move.to_square] = 1
    return result
