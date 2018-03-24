# multiAgents.py
# --------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

from util import manhattanDistance
from game import Directions
import random, util

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
    scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
    bestScore = max(scores)
    bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
    chosenIndex = random.choice(bestIndices) # Pick randomly among the best

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
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    "*** YOUR CODE HERE ***" 
    newGhostPositions = successorGameState.getGhostPositions()
    Food = currentGameState.getFood()
    Foodlist = Food.asList()
    GhostStates = currentGameState.getGhostStates()
    ScaredTimes = [ghostState.scaredTimer for ghostState in GhostStates]
    scaredTime = max(ScaredTimes)
    Capos = currentGameState.getCapsules()
    
    Gdistance = []
    for i in newGhostPositions:
        Gdistance.append(util.manhattanDistance(i, newPos))
    Gscore = min(Gdistance)
    
    Fscore = 0
    foodistance = util.PriorityQueue()
    if len(Foodlist) > 0:
        for i in Foodlist:
            foodistance.push((newPos,util.manhattanDistance(i, newPos)),util.manhattanDistance(i, newPos))
        fscore = foodistance.pop()[1]
    if fscore != 0:
        Fscore = (1.0 / fscore) *20
    else:
        Fscore = 1000
    
    Cscore =0
    capdistance = []
    if len(Capos) != 0:
        for i in Capos:
            capdistance.append(util.manhattanDistance(i, newPos))
        cscore = min(capdistance)
        if cscore != 0:
            Cscore = (1.0 / cscore) *20
        else:
            Cscore = 1000

    if scaredTime > 2 and scaredTime <= 40:
        return Cscore + Fscore + successorGameState.getScore()    

    if Gscore < 2:
        Gscore = -1000
        Cscore = 0
        Fscore = 0
    
    return Gscore + Cscore + Fscore + successorGameState.getScore()    

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

  def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
    self.index = 0 # Pacman is always agent index 0
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

      Directions.STOP:
        The stop direction, which is always legal

      gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

      gameState.getNumAgents():
        Returns the total number of agents in the game
    """
    "*** YOUR CODE HERE ***"
    def max_value(state, depth):
        depth += 1
        v = float('-Inf')
        if state.isWin() or state.isLose() or depth == self.depth:
            return self.evaluationFunction(state)
        for action in state.getLegalActions():
            if action != Directions.STOP: 
                v = max(v, min_value(state.generatePacmanSuccessor(action), depth, 1))
        return v
  
    def min_value(state, depth, numsofghost):
        v = float('Inf')
        if state.isWin() or state.isLose():
            return self.evaluationFunction(state)
        for action in state.getLegalActions(numsofghost):
            if numsofghost == (gameState.getNumAgents()-1):
                v = min(v, max_value(state.generateSuccessor(numsofghost, action), depth))
            else:
                v = min(v, min_value(state.generateSuccessor(numsofghost, action), depth, numsofghost + 1))
        return v
    
    Max = float('-Inf')
    for action in gameState.getLegalActions():
        if action != Directions.STOP:
            v = min_value(gameState.generatePacmanSuccessor(action), 0, 1)
            if v > Max:
                Max = v
                move = action
    return move  

class AlphaBetaAgent(MultiAgentSearchAgent):
  """
    Your minimax agent with alpha-beta pruning (question 3)
  """

  def getAction(self, gameState):
    """
      Returns the minimax action using self.depth and self.evaluationFunction
    """
    "*** YOUR CODE HERE ***"
    def max_value(state, depth, a, b):
        depth += 1
        v = float('-Inf')
        if state.isWin() or state.isLose() or depth == self.depth:
            return self.evaluationFunction(state)
        for action in state.getLegalActions():
            if action != Directions.STOP: 
                v = max(v, min_value(state.generatePacmanSuccessor(action), depth, 1, a, b))
                a = max(a, v)
                if a >= b:
                    return v           
        return v
  
    def min_value(state, depth, numsofghost, a, b):
        v = float('Inf')
        if state.isWin() or state.isLose():
            return self.evaluationFunction(state)
        for action in state.getLegalActions(numsofghost):
            if numsofghost == (gameState.getNumAgents()-1):
                v = min(v, max_value(state.generateSuccessor(numsofghost, action), depth, a, b))
                b = min(b, v)
                if a >= b:
                    return v
            else:
                v = min(v, min_value(state.generateSuccessor(numsofghost, action), depth, numsofghost + 1, a, b))
                b = min(b, v)
                if a >= b:
                    return v
        return v
    
    Max = float('-Inf')
    Min = float('Inf')
    for action in gameState.getLegalActions():
        if action != Directions.STOP:
            v = min_value(gameState.generatePacmanSuccessor(action), 0, 1, Max, Min)
            if v > Max:
                Max = v
                move = action
    return move  

  def getValue(self, gameState):
    """
      Returns the minimax action using self.depth and self.evaluationFunction
    """
    "*** YOUR CODE HERE ***"
    def max_value(state, depth, a, b):
        depth += 1
        v = float('-Inf')
        if state.isWin() or state.isLose() or depth == self.depth:
            return self.evaluationFunction(state)
        for action in state.getLegalActions():
            if action != Directions.STOP: 
                v = max(v, min_value(state.generatePacmanSuccessor(action), depth, 1, a, b))
                a = max(a, v)
                if a >= b:
                    return v           
        return v
  
    def min_value(state, depth, numsofghost, a, b):
        v = float('Inf')
        if state.isWin() or state.isLose():
            return self.evaluationFunction(state)
        for action in state.getLegalActions(numsofghost):
            if numsofghost == (gameState.getNumAgents()-1):
                v = min(v, max_value(state.generateSuccessor(numsofghost, action), depth, a, b))
                b = min(b, v)
                if a >= b:
                    return v
            else:
                v = min(v, min_value(state.generateSuccessor(numsofghost, action), depth, numsofghost + 1, a, b))
                b = min(b, v)
                if a >= b:
                    return v
        return v
    
    Max = float('-Inf')
    Min = float('Inf')
    for action in gameState.getLegalActions():
        if action != Directions.STOP:
            v = min_value(gameState.generatePacmanSuccessor(action), 0, 1, Max, Min)
            if v > Max:
                Max = v
    return Max 

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
    "*** YOUR CODE HERE ***" 
    def max_value(state, depth):
        depth += 1
        v = float('-Inf')
        if state.isWin() or state.isLose() or depth == self.depth:
            return self.evaluationFunction(state)
        for action in state.getLegalActions():
            if action != Directions.STOP: 
                v = max(v, min_value(state.generatePacmanSuccessor(action), depth, 1))
        return v
  
    def min_value(state, depth, numsofghost):
        v = float('Inf')
        if state.isWin() or state.isLose():
            return self.evaluationFunction(state)
        for action in state.getLegalActions(numsofghost):
            endlist = []
            if numsofghost == (gameState.getNumAgents()-1):
                v = max_value(state.generateSuccessor(numsofghost, action), depth)
                endlist.append(v)
                endlist = sorted(endlist)
            else:
                v = min_value(state.generateSuccessor(numsofghost, action), depth, numsofghost + 1)
                endlist.append(v)
                endlist = sorted(endlist)
        if len(endlist) > 1:
            return endlist[1]
        else:
            return endlist[0]
    
    Max = float('-Inf')
    for action in gameState.getLegalActions():
        if action != Directions.STOP:
            v = min_value(gameState.generatePacmanSuccessor(action), 0, 1)
            if v > Max:
                Max = v
                move = action
    return move    

def betterEvaluationFunction(currentGameState):
  """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
  """
  "*** YOUR CODE HERE ***"

  Food = currentGameState.getFood()
  Foodlist = Food.asList()
  GhostPos = currentGameState.getGhostPositions()
  GhostStates = currentGameState.getGhostStates()
  ScaredTimes = [ghostState.scaredTimer for ghostState in GhostStates]
  scaredTime = min(ScaredTimes)
  Capos = currentGameState.getCapsules()
  curPos = currentGameState.getPacmanPosition()
  
  Gscore = 0
  Gdistance = []
  for i in GhostPos:
      Gdistance.append(util.manhattanDistance(i, curPos))
  gscore = min(Gdistance)
  if gscore == 0:
      Gscore = -(10000 + len(Foodlist)*50)
  else:
      Gscore = -(1000/ gscore + len(Foodlist)*10)
      
  Fscore = 0
  fscore = 0
  foodistance = []
  if len(Foodlist) > 0 :
      for i in Foodlist:
          foodistance.append(util.manhattanDistance(i, curPos))
      fscore = min(foodistance)
      fscore = 500/fscore
      Fscore = 100000 - len(Foodlist)*1500
  else:
      Fscore = 100000
     
  Cscore = 0
  cscore = 0
  capdistance = []
  if len(Capos) != 0:
      for i in Capos:
          capdistance.append(util.manhattanDistance(i, curPos))
      cscore = min(capdistance)
      cscore = (500/ cscore) 
      Cscore = 5000 - len(Capos)*2000
  else:
      Cscore = 10000

  if scaredTime > 3 and scaredTime <= 37:
      return Fscore + currentGameState.getScore() + fscore + cscore + Cscore
          
  score = Gscore + Cscore + Fscore + currentGameState.getScore() + fscore + cscore
  return score

# Abbreviation
better = betterEvaluationFunction

class ContestAgent(MultiAgentSearchAgent):
  """
    Your agent for the mini-contest
  """

  def getAction(self, gameState):
    """
      Returns an action.  You can use any method you want and search to any depth you want.
      Just remember that the mini-contest is timed, so you have to trade off speed and computation.

      Ghosts don't behave randomly anymore, but they aren't perfect either -- they'll usually
      just make a beeline straight towards Pacman (or away from him if they're scared!)
    """
    "*** YOUR CODE HERE ***"
    def max_value(state, depth, a, b):
        depth += 1
        v = float('-Inf')
        if state.isWin() or state.isLose() or depth == 3:
            return better(state)
        for action in state.getLegalActions():
            if action != Directions.STOP: 
                v = max(v, min_value(state.generatePacmanSuccessor(action), depth, 1, a, b))
                a = max(a, v)
                if a >= b:
                    return v           
        return v
  
    def min_value(state, depth, numsofghost, a, b):
        v = float('Inf')
        if state.isWin() or state.isLose():
            return better(state)
        for action in state.getLegalActions(numsofghost):
            if numsofghost == (gameState.getNumAgents()-1):
                v = min(v, max_value(state.generateSuccessor(numsofghost, action), depth, a, b))
                b = min(b, v)
                if a >= b:
                    return v
            else:
                v = min(v, min_value(state.generateSuccessor(numsofghost, action), depth, numsofghost + 1, a, b))
                b = min(b, v)
                if a >= b:
                    return v
        return v
    
    Max = float('-Inf')
    Min = float('Inf')
    for action in gameState.getLegalActions():
        if action != Directions.STOP:
            v = min_value(gameState.generatePacmanSuccessor(action), 0, 1, Max, Min)
            if v > Max:
                Max = v
                move = action
    return move  