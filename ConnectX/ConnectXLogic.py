import numpy as np

class Board():
    def __init__(self, height=6, width=7):
        self.width = width
        self.height = height
        #Create empty board
        self.pieces = np.zeros(width*height).reshape(height, width)
    
    def __getitem__(self, index):
        return self.pieces(index)
    
    def get_legal_moves(self, color):
        moves = set()
        for x in range(self.width):
            for y in range(self.height):
                if self.pieces[y,x] == 0:
                    if not x in moves:
                        moves.add((x))
        return list(moves)
    
    def has_legal_moves(self):
        return 0 in self.pieces
    
    def is_win(self, color):
        #Check for vertical lines
        for x in range(self.width):
            count = 0
            for y in range(self.height):
                if np.swapaxes(self.pieces, 0, 1)[x,y] == color:
                    count += 1
                else:
                    count = 0
                if count == 4:
                    return True

        # Check for horizontal lines
        for y in range(self.height):
            for x in range(self.width):
                if self.pieces[y,x] == color:
                    count += 1
                else:
                    count = 0
                if count == 4:
                    return True

        # Check Negative Diagonals
        starts = [[5,3,4], [5,4,5], [5,5,6], [5,6,6], [4,6,5], [3,6,4]]
        for start in starts:
            count = 0
            for i in range(start[2]):
                if self.pieces[start[0]-i, start[1]-i] == color:
                    count += 1
                else:
                    count = 0
                if count == 4:
                    return True
        # Check Positive Diagonals
        starts = [[3,0,4], [4,0,5], [5,0,6], [5,1,6], [5,2,5], [5,3,4]]
        for start in starts:
            count = 0
            for i in range(start[2]):
                if self.pieces[start[0]-i, start[1]+i] == color:
                    count += 1
                else:
                    count = 0
                if count == 4:
                    return True

        return False

    def execute_move(self, move, color):
        for y in range(self.height):
            if self.pieces[self.height-y-1,move] == 0:
                assert self.pieces[self.height-y-1, move] == 0
                self.pieces[self.height-y-1,move] = color
                break

            