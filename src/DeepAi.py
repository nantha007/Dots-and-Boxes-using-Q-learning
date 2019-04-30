#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import random
import csv
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import model_from_json

##---------------------------------------------------
## class for AI
class deepai:
    def __init__(self, game, grid_size, alpha, epsilon, gamma, maxEpsilon, minEpsilon, total_episodes):
        self.game = game
        self.alpha = alpha
        self.action_size = 1
        self.epsilon = maxEpsilon
        self.grid_size = grid_size
        self.gamma = gamma
        self.maxEpsilon = maxEpsilon
        self.minEpsilon = minEpsilon
        self.episodeNum = total_episodes
        self.state_size = int(((2*grid_size + 1)**2-1)/2)
        self.layer_size = self.state_size*2
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(self.layer_size, input_dim=self.state_size, activation='relu'))
        model.add(Dense(self.layer_size, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model


    def write2csv(self):
        self.model.save_weights("../Results/network_" + str(self.grid_size) + ".h5")

    def loadFromCsv(self):
        self.model.load_weights("../Results/network_" + str(self.grid_size) + ".h5")


    def trainFromEpisode(self ):
        decayRate = (self.maxEpsilon - self.minEpsilon) / self.episodeNum
        for episode in range(self.episodeNum):
            self.epsilon = decayRate*self.epsilon
            # if (episode % 10000 == 0): print('Completed ' + str(episode) + ' episodes')
            self.learnFromEpisode()
        print("Training Done")


    def learnFromEpisode(self):
        NewGame = self.game(self.grid_size)
        _, move = self.getMove(NewGame)
        while move:
            move = self.trainAI(NewGame, move)
            # print(move)

    def trainAI(self, game, move):
        boxTaken = game.makeMove(move, 0, 0)
        count = 0
        if game.winner == game.player:
            count = 1
        reward = self.giveReward(game, boxTaken +  count)
        cummulativeReward = reward
        nextReward = 0.0
        selectedMove = None
        if ( not game.isBoardFull() ):
            bestNextMove, selectedMove = self.getMove(game)
            nextReward = self.model.predict(self.move2Array(bestNextMove))
        currentQValue = self.model.predict(self.move2Array(move))
        # currentQValue[0][]
        cummulativeReward = reward + self.gamma*(nextReward)
        target_f = currentQValue + cummulativeReward
        self.model.fit(self.move2Array(move), target_f, epochs=1, verbose=0)
        # self.qTable[move] = currentQValue + self.alpha * (cummulativeReward - currentQValue)
        return selectedMove

    def getMoveVsHuman(self):
        posMoves = self.possibleMoves(self.game)
        ## exploitation
        move = self.maxExploit(posMoves)
        self.game.makeMove( move,0,0 )

    def getMove(self, game):
        # print('getmove')
        posMoves = self.possibleMoves(game)
        ## exploitation
        move = self.maxExploit(posMoves)
        randomMove = move
        ## exploration
        if random.random() < self.epsilon:
            randomMove = random.choice(self.possibleMoves)
        return (move, randomMove)

    def possibleMoves(self, game):
        posStates = list()
        for i in range(0, int(((2*self.grid_size + 1)**2-1)/2)):
            if game.states[i] == ' ':
                tempBoard = game.states[:i] + str(1) + game.states[i+1:]
                posStates.append(tempBoard)
        return posStates

    def move2Array(self, states):
        array = np.zeros((1,len(states)))
        # print(states)
        for i in range(0,len(states)):
            if states[i] == ' ':
                array[0][i] = 0
            else:
                array[0][i] = 1
        return array

    def maxExploit(self, posStates):
        max = 0
        id = 0
        # print(posStates)
        for i in range(0,len(posStates)):
            currentQValue = self.model.predict(self.move2Array(posStates[i]))
            if currentQValue > max:
                id = i
                max = currentQValue
        state = posStates[id]
        return state

    def giveReward(self, game, mode):
        if mode==1:
            return 1.0
        elif mode == 2:
            return 5.0
        else:
            return 0.0
