# By
# Afnan Hussain     21L-5693    BDS-6A
# Alaiba Nawaz      21L-5650    BDS-6A

import numpy as np
import random as random



first_move = True  # Flag to track if it's the first move of AI
char = 'A'
edges = [' -- ', ' | ']
nodes = []
selected_edges = set()
turn = 1 # 1 for Player and -1 for AI
SCORE_AI = 0
SCORE_PLAYER = 0
BOX_CHECK = 0
NUM = None
transposition_table = {} # Initialize a transposition table

def find_index_and_mark(node1, node2, board):
    global selected_edges
    # Convert node names to indices
    row1 = int(node1[1])
    col1 = int(node1[2])
    row2 = int(node2[1])
    col2 = int(node2[2])

    if row1 == row2:
        # Calculate index between them
        index_between = row1*2, col1+col2

        # Place "--" on the found index
        board[index_between[0]][index_between[1]] = ' -- '
    else:
        # Calculate index between them
        index_between = row1+row2, col1*2

        # Place " | " on the found index
        
        board[index_between[0]][index_between[1]] = ' | '
       

    return board

def print_board(board):
    # Printing the board
    for i in range(len(board)):
        for j in range(len(board[i])):
                if board[i][j] == 'v':
                    print("    ",end="")
                elif board[i][j] == 'h':
                    print('    ',end=" ")
                elif board[i][j] in ['P','A']:
                    print(f' {board[i][j]}  ',end=" ")
                else:
                    print(board[i][j], end=' ')
        print()



def initialize_board():
    # Initialize the board with nodes and indices for edges
    board = [['.' if (i % 2 == 0 and j % 2 == 0) else 'h' if (i % 2 == 0) else 'v' if (
            j % 2 == 0) else '    ' for j in range(n * 2 - 1)] for i in range(m * 2 - 1)]

    # Fill in characters
    for i in range(m * 2 - 1):
        for j in range(n * 2 - 1):
            if j % 2 == 0 and i % 2 == 0:
                letter = char + str(i // 2) + str(j // 2)
                board[i][j] = letter
                
                nodes.append(letter)

    return board


def check_boxes_completed(board,turn):
    boxes_completed = 0
    for i in range(1, len(board), 2):
        for j in range(1, len(board[i]), 2):
            if board[i][j] == '    ':
                if board[i-1][j] != 'h' and board[i+1][j] != 'h' and board[i][j-1] != 'v' and board[i][j+1] != 'v':
                    if turn == 1:
                        board[i][j] = 'P'
                    if turn == -1:
                        board[i][j] = 'A'
    
    for i in range(1, len(board), 2):
        for j in range(1, len(board[i]), 2):
            if board[i][j] in ['P','A']:
                boxes_completed += 1
    
    return boxes_completed


def check_box_and_index(board,turn):
    index1=0
    index2=0
    for i in range(1, len(board), 2):
        for j in range(1, len(board[i]), 2):
            if board[i][j] == '  ':
                if board[i-1][j] != 'h' and board[i+1][j] != 'h' and board[i][j-1] != 'v' and board[i][j+1] != 'v':
                    if turn == 1:
                        board[i][j] = 'P'
                    if turn == -1:
                        board[i][j] = 'A'
                    index1 = i
                    index2 = j
    
    return index1,index2


def evaluate(board):
    # Simple evaluation function that counts the number of completed boxes for each player
    score_ai = np.count_nonzero(board == 'A')
    score_p1 = np.count_nonzero(board == 'P')
    return score_ai - score_p1


def game_over(board):
    selected_edges = 0
    for i in range(len(board)):
        for j in range(len(board[i])):
            if board[i][j] in edges:
                selected_edges += 1
    if selected_edges == (m-1)*n + m*(n-1):
        return True
    return False
    

def ai_score(board):
    score_ai = np.count_nonzero(board == 'A')
    return score_ai


def p_score(board):
    score_p = np.count_nonzero(board == 'P')
    return score_p


def minimax(board, depth, alpha, beta, is_maximizing):
    # Check if the current board state has been evaluated before
    board_key = tuple(map(tuple, board))
    if (board_key, depth) in transposition_table:
        return transposition_table[(board_key, depth)]

    if depth == 0 or game_over(board):
        return evaluate(board)

    if is_maximizing:
        max_eval = float('-inf')
        for i in range(0, len(board)):
            for j in range(0, len(board[i])):
                if board[i][j] == 'h' or board[i][j] == 'v':
                    new_board = np.copy(board)
                    node1, node2 = get_nodes_from_edge(board, i, j)
                    score_ai = ai_score(new_board)
                    new_board = find_index_and_mark(node1, node2, new_board)
                    check_boxes_completed(new_board, -1)
                    score_ai_new = ai_score(new_board)
                    if score_ai_new > score_ai:
                        eval_score = minimax(new_board, depth-1, alpha, beta, True)
                    else:
                        eval_score = minimax(new_board, depth-1, alpha, beta, False)
                    max_eval = max(max_eval, eval_score)
                    alpha = max(alpha, eval_score)
                    if beta <= alpha:
                        break
        transposition_table[(board_key, depth)] = max_eval
        return max_eval
    else:
        min_eval = float('inf')
        for i in range(0, len(board)):
            for j in range(0, len(board[i])):
                if board[i][j] == 'h' or board[i][j] == 'v':
                    new_board = np.copy(board)
                    node1, node2 = get_nodes_from_edge(board, i, j)
                    score_p = p_score(new_board)
                    new_board = find_index_and_mark(node1, node2, new_board)
                    check_boxes_completed(new_board, 1)
                    score_p_new = p_score(new_board)
                    if score_p_new == score_p:
                        eval_score = minimax(new_board, depth-1, alpha, beta, True)
                    else:
                        eval_score = minimax(new_board, depth-1, alpha, beta, False)
                    min_eval = min(min_eval, eval_score)
                    beta = min(beta, eval_score)
                    if beta <= alpha:
                        break
        transposition_table[(board_key, depth)] = min_eval
        return min_eval


def get_nodes_from_edge(board, i, j):
    if board[i][j] == 'h':
        node1 = board[i][j-1]
        node2 = board[i][j+1]
    else:
        node1 = board[i-1][j]
        node2 = board[i+1][j]
    return node1, node2


def available_no_moves(board):
    count_h = np.count_nonzero(board == 'h')
    count_v = np.count_nonzero(board == 'v')

    return count_h + count_v


def calculate_depth(no_moves,total_moves=18000):
    depth = 0
    while total_moves >= no_moves:
        total_moves //= no_moves
        depth += 1
    return depth


def ai_make_move(board, first_move):
    best_score = float('-inf')
    best_move = None
    no_moves = available_no_moves(board)
    if no_moves > 1:
        depth = calculate_depth(no_moves)
    else:
        depth = 1
    if first_move == True and n > 3:
        index = random_move(board)
        if index[2] =='h':
            node1 = board[index[0]][index[1]-1]
            node2 = board[index[0]][index[1]+1]
        else:
            node1 = board[index[0]-1][index[1]]
            node2 = board[index[0]+1][index[1]]
        best_move=(node1,node2)
    else:
        for i in range(0, len(board)):
            for j in range(0, len(board[i])):
                if board[i][j] == 'h' or board[i][j] == 'v':
                    new_board = np.copy(board)
                    node1, node2 = get_nodes_from_edge(board, i, j)
                    if (node1, node2) in selected_edges or (node2, node1) in selected_edges:
                        continue
                    score_ai = ai_score(new_board)
                    new_board = find_index_and_mark(node1, node2, new_board)
                    check_boxes_completed(new_board, -1)
                    score_ai_new = ai_score(new_board)
                    if score_ai_new > score_ai:
                            
                            eval_score = minimax(new_board, depth, float('-inf'), float('inf'), True)
                    else:
                            
                            eval_score = minimax(new_board, depth, float('-inf'), float('inf'), False)
                    if eval_score > best_score:
                        best_score = eval_score
                        best_move = (node1, node2)
    return best_move
    

def random_move(board):
    available_moves = []
    for i in range(0, len(board)):
        for j in range(0, len(board[i])):
            if board[i][j] in ['h', 'v']:
                available_moves.append((i, j, board[i][j]))
    return random.choice(available_moves)


def check_move(node1,node2):
    if node1 == node2:
        return "Both nodes cannot be same"
    if node1 not in nodes:
        return f'Node: {node1} not valid node. Enter Correct Nodes!'
    if node2 not in nodes:
        return f'Node: {node2} not valid node. Enter Correct Nodes!'
    elif (node1, node2) in selected_edges or (node2, node1) in selected_edges:
        return "Edge already selected. Choose another edge."
    row1 = int(node1[1])
    col1 = int(node1[2])
    row2 = int(node2[1])
    col2 = int(node2[2])

    if abs(row1 - row2)>1 or abs(col1-col2)>1:
        return "Nodes are not adjacent. Choose other nodes."
    if abs(row1 - row2)>=1 and col1!=col2:
        return "Nodes are not adjacent. Choose other nodes."
    
    return 'valid'


    


print("Welcome to Dots and Boxes Game\n")

# Loop until a valid integer input is provided
while not isinstance(NUM, int) or NUM <= 0 or NUM > 10:
    try:
        NUM = int(input("Enter n matrix size (nxn) (max size = 10): "))
        if NUM <= 0:
            print("Please enter a positive integer.")
        elif NUM > 10:
            print("Please enter a number less than 10.")
    except ValueError:
        print("Invalid input. Please enter an integer.")

m = NUM
n = NUM
board = initialize_board()


while game_over(board)!=True:
    print('----------------------------')
    print_board(board)
    print('\n')
    print(f"Your Score: {SCORE_PLAYER} \nAI Score: {SCORE_AI}\n")
    if turn == 1:
        print(f"Player {1} turn")
        node1 = input("Enter first node: ").upper()
        node2 = input("Enter second node: ").upper()
        check = check_move(node1,node2)
        if check != 'valid':
            print(check)
            continue
        
        board = find_index_and_mark(node1, node2, board)
        # Add the selected edge to the set
        selected_edges.add((node1, node2))
        boxes_completed = check_boxes_completed(board, turn)
        if boxes_completed > BOX_CHECK:
            print(f"{boxes_completed} box(es) completed.")
            DIFF = boxes_completed - BOX_CHECK
            SCORE_PLAYER += DIFF
            BOX_CHECK = boxes_completed
            turn *= -1
    else:
        print("AI's turn")
        new_board = np.copy(board)
        ai_move = ai_make_move(new_board,first_move)
        first_move = False
        if ai_move:
            node1, node2 = ai_move
            board = find_index_and_mark(node1, node2, board)
            selected_edges.add((node1, node2))
            print(f"AI made a move: {node1}, {node2}")
            boxes_completed = check_boxes_completed(board, turn)
            if boxes_completed > BOX_CHECK:
                print(f"{boxes_completed} box(es) completed.")
                DIFF = boxes_completed - BOX_CHECK
                SCORE_AI += DIFF
                BOX_CHECK = boxes_completed
                turn *= -1
        else:
            print("AI couldn't find a valid move.")
    turn *= -1
print('\n')

# Printing the final board and score
if SCORE_PLAYER == SCORE_AI:
    print("ITS A TIE!")
elif SCORE_AI < SCORE_PLAYER:
    print("YOU WON! CONGRATS")
else:
    print("AI SHEHZADAYY WON!!")
print("\nFinal Score")
print(f"Your Score: {SCORE_PLAYER} \nAI Score: {SCORE_AI}\n")
print("\nFinal board:")
print_board(board)