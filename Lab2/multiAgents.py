# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random
import util

from game import Agent


class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(
            gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(
            len(scores)) if scores[index] == bestScore]
        # Pick randomly among the best
        chosenIndex = random.choice(bestIndices)

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [
            ghostState.scaredTimer for ghostState in newGhostStates]

        closestghost = min(
            [manhattanDistance(newPos, ghost.getPosition()) for ghost in newGhostStates])

        if closestghost:
            ghost_dist = -10/closestghost
        else:
            ghost_dist = -1000

        foodList = newFood.asList()
        if foodList:
            closestfood = min([manhattanDistance(newPos, food)
                               for food in foodList])
        else:
            closestfood = 0

        # large weight to number of food left
        return (-2 * closestfood) + ghost_dist - (100*len(foodList))


def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """

        def maxAgent(depth, agentIndex, state):
            bestScore = -float('inf')
            bestAction = None
            legalActions = state.getLegalActions(
                agentIndex)  # Legal moves of Pacman

            # If there are no legal moves, wether game has finished or Pacman has lost
            if len(legalActions) == 0 or state.isWin() or state.isLose():
                return self.evaluationFunction(state)

            for action in legalActions:
                # Get successors of current state
                successor = state.generateSuccessor(agentIndex, action)
                score = minAgent(depth, 1, successor)
                if (score > bestScore):
                    bestScore = score
                    bestAction = action

            if depth == 0:
                return bestAction
            else:
                return bestScore

        def minAgent(depth, agentIndex, state):
            bestScore = float('inf')
            legalActions = state.getLegalActions(agentIndex)

            if len(legalActions) == 0 or state.isWin() or state.isLose():
                return self.evaluationFunction(state)

            if agentIndex == state.getNumAgents()-1:  # Last ghost
                for action in legalActions:
                    successor = state.generateSuccessor(agentIndex, action)
                    if depth == self.depth-1:  # Last layer
                        score = self.evaluationFunction(successor)
                    else:
                        score = maxAgent(depth+1, 0, successor)
                    bestScore = min(bestScore, score)
            else:  # Not last ghost
                for action in legalActions:
                    successor = state.generateSuccessor(agentIndex, action)
                    score = minAgent(depth, agentIndex+1, successor)
                    bestScore = min(bestScore, score)
            return bestScore

        return maxAgent(0, 0, gameState)


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        def maxAgent(depth, agentIndex, alpha, beta, state):
            bestScore = -float('inf')
            bestAction = None
            legalActions = state.getLegalActions(
                agentIndex)  # Legal moves of Pacman

            # If there are no legal moves, wether game has finished or Pacman has lost
            if len(legalActions) == 0 or state.isWin() or state.isLose():
                return self.evaluationFunction(state)

            for action in legalActions:
                # Get successors of current state
                successor = state.generateSuccessor(agentIndex, action)
                score = minAgent(depth, 1, alpha, beta, successor)
                alpha = max(alpha, score)

                if (score > bestScore):
                    bestScore = score
                    bestAction = action
                if (alpha > beta):
                    break

            if depth == 0:
                return bestAction
            else:
                return bestScore

        def minAgent(depth, agentIndex, alpha, beta, state):
            bestScore = float('inf')
            legalActions = state.getLegalActions(agentIndex)

            if len(legalActions) == 0:
                return self.evaluationFunction(state)

            if agentIndex == state.getNumAgents()-1:  # Last ghost
                for action in legalActions:
                    successor = state.generateSuccessor(agentIndex, action)

                    if depth == self.depth-1:  # Last layer
                        score = self.evaluationFunction(successor)
                    else:
                        score = maxAgent(depth+1, 0, alpha, beta, successor)

                    beta = min(beta, score)
                    bestScore = min(bestScore, score)

                    if (beta < alpha):
                        break
            else:  # Not last ghost
                for action in legalActions:
                    successor = state.generateSuccessor(agentIndex, action)
                    score = minAgent(depth, agentIndex+1,
                                     alpha, beta, successor)
                    beta = min(beta, score)
                    bestScore = min(bestScore, score)

                    if (beta < alpha):
                        break
            return bestScore

        alpha = -float('inf')
        beta = float('inf')

        return maxAgent(0, 0, alpha, beta, gameState)


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        def maxAgent(depth, agentIndex, state):
            bestScore = -float('inf')
            bestAction = None
            legalActions = state.getLegalActions(
                agentIndex)  # Legal moves of Pacman

            # If there are no legal moves, wether game has finished or Pacman has lost
            if len(legalActions) == 0 or state.isWin() or state.isLose():
                return self.evaluationFunction(state)

            for action in legalActions:
                # Get successors of current state
                successor = state.generateSuccessor(agentIndex, action)
                score = minAgent(depth, agentIndex+1, successor)

                if (score > bestScore):
                    bestScore = score
                    bestAction = action

            if depth == 0:
                return bestAction
            else:
                return bestScore

        def minAgent(depth, agentIndex, state):
            legalActions = state.getLegalActions(agentIndex)
            if len(legalActions) == 0:
                return self.evaluationFunction(state)

            score = 0
            if agentIndex == state.getNumAgents()-1:  # Last ghost
                for action in legalActions:
                    successor = state.generateSuccessor(agentIndex, action)

                    if depth == self.depth-1:  # Last layer
                        score += (1.0/len(legalActions)) * \
                            self.evaluationFunction(successor)
                    else:
                        score += (1.0/len(legalActions)) * \
                            maxAgent(depth+1, 0, successor)
            else:  # Not last ghost
                for action in legalActions:
                    successor = state.generateSuccessor(agentIndex, action)
                    score += (1.0/len(legalActions)) * \
                        minAgent(depth, agentIndex+1, successor)
            return score

        return maxAgent(0, 0, gameState)


def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newCapsules = currentGameState.getCapsules()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    closestGhost = min([manhattanDistance(newPos, ghost.getPosition())
                        for ghost in newGhostStates])

    if newCapsules:
        closestCapsule = min([manhattanDistance(newPos, caps)
                              for caps in newCapsules])
    else:
        closestCapsule = 0

    if closestCapsule:
        closest_capsule = -3 / closestCapsule
    else:
        closest_capsule = 100

    if closestGhost:
        ghost_distance = -2 / closestGhost
    else:
        ghost_distance = -500

    foodList = newFood.asList()
    if foodList:
        closestFood = min([manhattanDistance(newPos, food)
                           for food in foodList])
    else:
        closestFood = 0

    return -2 * closestFood + ghost_distance - 10 * len(foodList) + closest_capsule


# Abbreviation
better = betterEvaluationFunction
