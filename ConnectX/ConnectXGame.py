from __future__ import print_function
import sys
sys.path.append('..')
from Game import Game
from .ConnectXLogic import Board
import numpy as np

class ConnectXGame(Game):
    def __init__(self, n=4, width=7, height=6):
        self.n=n
        self.height=height
        self.width=width

    def getInitBoard(self):
        b = Board(self.height, self.width)
        return b.pieces
    
    def getBoardSize(self):
        return (self.height, self.width)
    
    def getActionSize(self):
        return self.width

    def getNextState(self, board, player, action):
        b = Board(self.height, self.width)
        b.pieces = np.copy(board)
        b.execute_move(action, player)
        return (b.pieces, -player)
    
    def getValidMoves(self, board, player):
        valids = np.zeros(self.getActionSize())
        b = Board(self.height, self.width)
        b.pieces = np.copy(board)
        legalMoves =  b.get_legal_moves(player)
        for x in legalMoves:
            valids[x]=1
        #print(np.array(valids))
        return np.array(valids)

    def getGameEnded(self, board, player):
        # Return Tuple (game ended, player score) where game ended is True or False
        b = Board(self.height, self.width)
        b.pieces = np.copy(board)

        if b.is_win(player):
            return 1
        if b.is_win(-player):
            return -1
        if b.has_legal_moves():
            return 0
        # Draw
        return 1e-4

    def getCanonicalForm(self, board, player):
        # return state if player==1, else return -state if player==-1
        return player*board
    
    def getSymmetries(self, board, pi):
        assert(len(pi) == self.width)
        l = []
        pi_board = np.reshape(pi, (self.width))
        pi_board = np.expand_dims(pi_board, -1)
        for i in [False, True]:
            newPi = pi_board
            newB = board
            if i:
                newB = np.fliplr(newB)
                newPi = np.fliplr(newPi)

            l += [(newB, list(newPi.ravel()))]
        return l
    
    def stringRepresentation(self, board):
        # 7x6 numpy array (canonical board)
        return board.tostring()

    @staticmethod
    def display(board):
        print(" -----------------------")
        print(' '.join(map(str, range(len(board[0])))))
        print(board)
        print(" -----------------------")
    