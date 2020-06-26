# https://medium.com/byte-tales/the-classic-tic-tac-toe-game-in-python-3-1427c68b8874
# Implementation of Two Player Tic-Tac-Toe game in Python.

''' We will make the board using dictionary
    in which keys will be the location(i.e : top-left,mid-right,etc.)
    and initialliy it's values will be empty space and then after every move
    we will change the value according to player's choice of move. '''
import math
import time
from random import random
from threading import Thread

from NeuralNetwork.NeuralNetwork import NeuralNetwork, LoadFromFile

theBoard = {'7': ' ', '8': ' ', '9': ' ',
            '4': ' ', '5': ' ', '6': ' ',
            '1': ' ', '2': ' ', '3': ' '}

board_keys = []
neural_net1 = LoadFromFile("neural_net1.pkl")
if not neural_net1:
    neural_net1 = NeuralNetwork(27, 9, 4, 9)
neural_net2 = LoadFromFile("neural_net2.pkl")
if not neural_net2:
    neural_net2 = NeuralNetwork(27, 9, 4, 9)

for key in theBoard:
    board_keys.append(key)

''' We will have to print the updated board after every move in the game and 
    thus we will make a function in which we'll define the printBoard function
    so that we can easily print the board everytime by calling this function. '''


def printBoard(board):
    print(board['7'] + '|' + board['8'] + '|' + board['9'])
    print('-+-+-')
    print(board['4'] + '|' + board['5'] + '|' + board['6'])
    print('-+-+-')
    print(board['1'] + '|' + board['2'] + '|' + board['3'])


# Now we'll write the main function which has all the gameplay functionality.
def game(x, o, x_net, o_net):
    turn = 'X'
    count = 0
    int_board_list_o = []
    int_board_list_x = []
    int_board = []

    neural_net_o = o_net
    neural_net_x = x_net

    while count <= 9:
        if x == 2 or o == 2:
            printBoard(theBoard)
            print("It's your turn," + turn + ".Move to which place?")

        move = None

        if turn == 'O':
            if o == 0:
                rand = int(random()*9+1)
                move = str(rand)
            if o == 1:
                int_board = []
                for place in theBoard.values():
                    if place == ' ':
                        int_board.append(1)
                        int_board.append(0)
                        int_board.append(0)
                    elif place == 'X':  # player 'X'
                        int_board.append(0)
                        int_board.append(1)
                        int_board.append(0)
                    else:  # AI 'O'
                        int_board.append(0)
                        int_board.append(0)
                        int_board.append(1)
                int_board_list_o.append(int_board)
                outputs = neural_net_o.GetOutputs(int_board)
                print(outputs)
                move = -1
                max_output = 0
                for index in range(len(outputs)):
                    if outputs[index] > max_output:
                        move = index
                        max_output = outputs[index]
                move = str(move+1)
                if theBoard[move] != ' ':
                    move = str(int(random()*9+1))
                    neural_net_x.CalculateSlopeValuesFromCost(int_board, -0.02)
            if o == 2:
                move = input()
        if turn == 'X':
            if x == 0:
                rand = int(random()*9+1)
                move = str(rand)
            if x == 1:
                int_board = []
                for place in theBoard.values():
                    if place == ' ':
                        int_board.append(1)
                        int_board.append(0)
                        int_board.append(0)
                    elif place == 'X':  # player 'X'
                        int_board.append(0)
                        int_board.append(1)
                        int_board.append(0)
                    elif place == 'O':
                        int_board.append(0)
                        int_board.append(0)
                        int_board.append(1)
                int_board_list_x.append(int_board)
                outputs = neural_net_x.GetOutputs(int_board)
                print(outputs)
                move = -1
                max_output = 0
                for index in range(len(outputs)):
                    if outputs[index] > max_output:
                        move = index
                        max_output = outputs[index]
                move = str(move + 1)
                if theBoard[move] != ' ':
                    move = str(int(random()*9+1))
                    neural_net_x.CalculateSlopeValuesFromCost(int_board, -0.02)
            if x == 2:
                move = input()

        if theBoard[move] == ' ':
            theBoard[move] = turn
            count += 1
        else:
            if x == 2 or o == 2:
                print("That place is already filled.\nMove to which place?")
            if turn == 'O':
                if o == 1:
                    int_board_list_o.pop(-1)
                    neural_net_o.CalculateSlopeValuesFromCost(int_board, -0.01)
            if turn == 'X':
                if x == 1:
                    int_board_list_x.pop(-1)
                    neural_net_x.CalculateSlopeValuesFromCost(int_board, -0.01)
            continue

        # Now we will check if player X or O has won,for every move after 5 moves.
        if count >= 5:
            if theBoard['7'] == theBoard['8'] == theBoard['9'] != ' ':  # across the top
                if x == 2 or o == 2:
                    printBoard(theBoard)
                    print("\nGame Over.\n")
                    print(" **** " + turn + " won. ****")
                break
            elif theBoard['4'] == theBoard['5'] == theBoard['6'] != ' ':  # across the middle
                if x == 2 or o == 2:
                    printBoard(theBoard)
                    print("\nGame Over.\n")
                    print(" **** " + turn + " won. ****")
                break
            elif theBoard['1'] == theBoard['2'] == theBoard['3'] != ' ':  # across the bottom
                if x == 2 or o == 2:
                    printBoard(theBoard)
                    print("\nGame Over.\n")
                    print(" **** " + turn + " won. ****")
                break
            elif theBoard['1'] == theBoard['4'] == theBoard['7'] != ' ':  # down the left side
                if x == 2 or o == 2:
                    printBoard(theBoard)
                    print("\nGame Over.\n")
                    print(" **** " + turn + " won. ****")
                break
            elif theBoard['2'] == theBoard['5'] == theBoard['8'] != ' ':  # down the middle
                if x == 2 or o == 2:
                    printBoard(theBoard)
                    print("\nGame Over.\n")
                    print(" **** " + turn + " won. ****")
                break
            elif theBoard['3'] == theBoard['6'] == theBoard['9'] != ' ':  # down the right side
                if x == 2 or o == 2:
                    printBoard(theBoard)
                    print("\nGame Over.\n")
                    print(" **** " + turn + " won. ****")
                break
            elif theBoard['7'] == theBoard['5'] == theBoard['3'] != ' ':  # diagonal
                if x == 2 or o == 2:
                    printBoard(theBoard)
                    print("\nGame Over.\n")
                    print(" **** " + turn + " won. ****")
                break
            elif theBoard['1'] == theBoard['5'] == theBoard['9'] != ' ':  # diagonal
                if x == 2 or o == 2:
                    printBoard(theBoard)
                    print("\nGame Over.\n")
                    print(" **** " + turn + " won. ****")
                break

                # If neither X nor O wins and the board is full, we'll declare the result as 'tie'.
        if count == 9:
            if x == 2 or o == 2:
                print("\nGame Over.\n")
                print("It's a Tie!!")
            turn = ""
            break

        # Now we have to change the player after every move.
        if turn == 'X':
            turn = 'O'
        else:
            turn = 'X'

    severity = 0.01

    if turn == 'O':
        if o == 1:
            for board in reversed(int_board_list_o):
                neural_net_o.CalculateSlopeValuesFromCost(board, severity)
            # print("Mutated O Win")
        if x == 1:
            for board in reversed(int_board_list_x):
                neural_net_x.CalculateSlopeValuesFromCost(board, -severity)
            # print("Mutated X Loss")
    elif turn == 'X':
        if o == 1:
            for board in reversed(int_board_list_o):
                neural_net_o.CalculateSlopeValuesFromCost(board, -severity)
            # print("Mutated O Loss")
        if x == 1:
            for board in reversed(int_board_list_x):
                neural_net_x.CalculateSlopeValuesFromCost(board, severity)
            # print("Mutated X Win")
    else:
        # for board in reversed(int_board_list):
        #     neural_net.CalculateSlopeValuesFromCost(board, 0.01)
        # print("Mutated Tie (considered minor win for ai)")
        pass

    # Now we will ask if player wants to restart the game or not.

    # restart = input("Do want to play Again?(y/n)")
    # if restart == "y" or restart == "Y":
    for key in board_keys:
        theBoard[key] = " "


def game_chunks(chunk_count, x, o):
    """

    :param chunk_count:
    :param x: 0 random , 1 neural_net1 , 2 user
    :param o: 0 random , 1 neural_net2 , 2 user
    :return:
    """
    o_net = neural_net1
    x_net = neural_net2

    for _ in range(chunk_count):
        game(x, o, x_net, o_net)

    if o == 1:
        o_net.MutateSlopeValues(0.01)
        if o_net is neural_net1:
            neural_net1.SaveToFile("neural_net1.pkl")
        elif o_net is neural_net2:
            neural_net2.SaveToFile("neural_net2.pkl")
    if x == 1:
        x_net.MutateSlopeValues(0.01)
        if x_net is neural_net1:
            neural_net1.SaveToFile("neural_net1.pkl")
        elif x_net is neural_net2:
            neural_net2.SaveToFile("neural_net2.pkl")


if __name__ == "__main__":
    timer_total = time.time()
    for _ in range(100):
        timer = time.time()

        # game_chunks(100, 1, 0)
        # game_chunks(100, 0, 1)
        game_chunks(100, 2, 1)

        print(f"100 games Twice took {time.time() - timer} seconds")
        print(_)
    print()
    print(f"10_000 chunks Twice took {time.time() - timer_total} seconds")
