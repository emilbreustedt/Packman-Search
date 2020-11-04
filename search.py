# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def generalGraphSearch(problem, structure):
    """
    Defines a general algorithm to search a graph.
    Parameters are structure, which can be any data structure with .push() and .pop() methods, and problem, which is the
    search problem.
    """
    # Initialise the list of finished nodes
    done = []

    # Push start node in the data structure: [(state, action taken, cost)]
    structure.push([(problem.getStartState(), "Stop", 0)])

    # While the structure is not empty
    while not structure.isEmpty():
        # pop path from data structure
        path = structure.pop()

        # current state is first element of last tuple of path
        curr_state = path[-1][0]

        # if current state  goal state
        if problem.isGoalState(curr_state):
            # return actions to the goal state
            return [x[1] for x in path][1:]

        # if current state is not visited
        if curr_state not in done:
            # mark current state as done
            done.append(curr_state)
            # for states after current states
            for next in problem.getSuccessors(curr_state):
                # if successor's state is unfinished
                if next[0] not in done:
                    # parent path
                    nextPath = path[:]
                    # append next node to parent path
                    nextPath.append(next)
                    # push next path into the data structure
                    structure.push(nextPath)
    # if fails, return false
    return False

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """
    # empty Stack
    stack = util.Stack()
    # DFS is generalGraphSearch with Stack
    return generalGraphSearch(problem, stack)

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    # empty Queue
    queue = util.Queue()
    # BFS is generalGraphSearch with Queue
    return generalGraphSearch(problem, queue)


def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    
    # The cost for UCS are backward costs
    # get the actions of the path without first Stop
    # get specific costs with  problem.getCostOfActions
    cost = lambda path: problem.getCostOfActions([x[1] for x in path][1:])

    # empty priority queue with backward costs
    priority_queue = util.PriorityQueueWithFunction(cost)

    # UCS is general graph search with generated priority queue
    return generalGraphSearch(problem, priority_queue)

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""

    # The cost for a* seach is f(x) = g(x) + h(x)
    # The backward cost defined in UCS (problem.getCostOfActions([x[1] for x in path][1:])) is g(x)
    # The heuristic is h(x), heuristic(state, problem),
    # where state = path[-1][0], which is the first element in the last tuple of the path
    cost = lambda path: problem.getCostOfActions([x[1] for x in path][1:]) + heuristic(path[-1][0], problem)

    # Construct an empty priority queue that sorts using f(x)
    pq = util.PriorityQueueWithFunction(cost)

    # A* is general graph search with the PriorityQueue sorting by the f(x) as the data structure
    return generalGraphSearch(problem, pq)


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
