# valueIterationAgents.py
# -----------------------
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


import mdp, util

from learningAgents import ValueEstimationAgent

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0

        for iteration in range(self.iterations):
            # For each iteration, we store the Qvalues of all the states in valuesIter
            valuesIter = util.Counter()
            for state in mdp.getStates():  
                # Code only for non-terminal states
                if not mdp.isTerminal(state): 
                    # BestAction has the best Action for a given state
                    bestAction = self.getAction(state)
                    if bestAction != None:
                        # The Qvalue for the given state and bestAcion to apply to this state
                        Qvalue = self.getQValue(state,bestAction)
                        # Store the value in the dictionary of the iteration
                        valuesIter[state] = Qvalue
            # When iteration finishes, store the values of the iteration into self.values
            self.values = valuesIter


    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """

        """
        Returns list of (nextState, prob) pairs
        representing the states reachable
        from 'state' by taking 'action' along
        with their transition probabilities.
        """

        # List of (nextState,prob) reachable from state applying action
        probActions = self.mdp.getTransitionStatesAndProbs(state,action)

        Qvalue = 0

        for adjState in probActions: 
            nextState = adjState[0]
            prob = adjState[1]
            
            # Formula for computing the Qvalue
            Qvalue += prob*(self.mdp.getReward(state, action, nextState) + self.discount * self.values[nextState])

        return Qvalue


    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """

        if self.mdp.isTerminal(state): 
            return None

        maxValue = -float('inf')

        # Obtain the legal actions for this state
        legalactions = self.mdp.getPossibleActions(state)

        for action in legalactions: 
            Qvalue = self.getQValue(state,action)
            # Obtaing the action whose value is the highest
            if (Qvalue > maxValue): 
                maxValue = Qvalue
                bestAction = action

        return bestAction
    

       # util.raiseNotDefined()

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)
