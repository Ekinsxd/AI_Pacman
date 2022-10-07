# baselineTeam.py
# ---------------
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


# baselineTeam.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html
from __future__ import print_function
from captureAgents import CaptureAgent
import distanceCalculator
import random, time, util, sys
from game import Directions
import game
from util import nearestPoint


#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first = 'NotReallyDefensiveReflexAgent', second = 'DefensiveReflexAgent'):
  """
  This function should return a list of two agents that will form the
  team, initialized using firstIndex and secondIndex as their agent
  index numbers.  isRed is True if the red team is being created, and
  will be False if the blue team is being created.

  As a potentially helpful development aid, this function can take
  additional string-valued keyword arguments ("first" and "second" are
  such arguments in the case of this function), which will come from
  the --redOpts and --blueOpts command-line arguments to capture.py.
  For the nightly contest, however, your team will be created without
  any extra arguments, so you should make sure that the default
  behavior is what you want for the nightly contest.
  """
  return [eval(first)(firstIndex), eval(second)(secondIndex)]

##########
# Agents #
##########

class ReflexCaptureAgent(CaptureAgent):
  """
  A base class for reflex agents that chooses score-maximizing actions.
  Code below is copied to ensure that the other classes that inherit
  this class works.
  """
 
  def registerInitialState(self, gameState):
    self.start = gameState.getAgentPosition(self.index)
    CaptureAgent.registerInitialState(self, gameState)

  def chooseAction(self, gameState):
    """
    Picks among the actions with the highest Q(s,a).
    """
    actions = gameState.getLegalActions(self.index)

    values = [self.evaluate(gameState, a) for a in actions]

    maxValue = max(values)
    bestActions = [a for a, v in zip(actions, values) if v == maxValue]

    foodLeft = len(self.getFood(gameState).asList())

    if foodLeft <= 2 or gameState.getAgentState(self.index).numCarrying > 2:
      bestDist = 9999
      for action in actions:
        successor = self.getSuccessor(gameState, action)
        pos2 = successor.getAgentPosition(self.index)
        dist = self.getMazeDistance(self.start,pos2)
        if dist < bestDist:
          bestAction = action
          bestDist = dist
      return bestAction

    return random.choice(bestActions)

  def getSuccessor(self, gameState, action):
    """
    Finds the next successor which is a grid position (location tuple).
    """
    successor = gameState.generateSuccessor(self.index, action)
    pos = successor.getAgentState(self.index).getPosition()
    if pos != nearestPoint(pos):
      # Only half a grid position was covered
      return successor.generateSuccessor(self.index, action)
    else:
      return successor

  def evaluate(self, gameState, action):
    """
    Computes a linear combination of features and feature weights
    """

    features = self.getFeatures(gameState, action)
    weights = self.getWeights(gameState, action)

    #if self.index == 1:
      #print(str(features) + str(weights), file=sys.stderr)
      # print(gameState.getAgentState(self.index)) # Print out a text representation of the world.

    return features * weights

  def getFeatures(self, gameState, action):
    """
    Returns a counter of features for the state
    """
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    features['successorScore'] = self.getScore(successor)

    return features

  def getWeights(self, gameState, action):
    """
    Normally, weights do not depend on the gamestate.  They can be either
    a counter or a dictionary.
    """
    return {'successorScore': 1.0}


class NotReallyDefensiveReflexAgent(ReflexCaptureAgent):
  """
  This agents sort of acts as the defensive agent, but can
  also go on the offensive if no invaders are spotted. However,
  it will prefer chasing enemy-Pacmans.
  """

  def getFeatures(self, gameState, action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)

    myState = successor.getAgentState(self.index)
    myPos = myState.getPosition()

    # Computes whether we're on defense (1) or offense (0)
    features['onDefense'] = 1

    # Computes distance to invaders we can see
    enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
    invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
    enmy_ghost = [a for a in enemies if not a.isPacman and a.getPosition() != None]

    #If there isn't any invaders in sight, go offensive.
    features['numInvaders'] = len(invaders)
    if len(invaders) > 0:
      dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
      features['invaderDistance'] = min(dists)
    else:
        features['onDefense'] = 0

    
    #These are needed, or else the agent will stay around spawn.
    if action == Directions.STOP: features['stop'] = 100
    rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
    if action == rev: features['reverse'] = 1

    foodList = self.getFood(successor).asList()    
    features['successorScore'] = -len(foodList)

    # Compute distance to the nearest food
    if len(foodList) > 0: # This should always be True,  but better safe than sorry
      myPos = successor.getAgentState(self.index).getPosition()
      minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
      features['distanceToFood'] = minDistance
      if len(enmy_ghost) > 0:
        features['distanceToFood'] = 0
    

    # Determine if the enemy is closer to you than they were last time
    # and you are in their territory.
    close_dist = 9999.0
    if self.index == 1 and gameState.getAgentState(self.index).isPacman:
      opp_fut_state = [successor.getAgentState(i) for i in self.getOpponents(successor)]
      chasers = [p for p in opp_fut_state if p.getPosition() != None and not p.isPacman]
      if len(chasers) > 0:
        close_dist = min([float(self.getMazeDistance(myPos, c.getPosition())) for c in chasers])

      # View the action and close distance information for each 
      # possible move choice.
      features['fleeEnemy'] = 1/close_dist

    return features

  def getWeights(self, gameState, action):
    return {'numInvaders': -1000, 'onDefense': 100, 'invaderDistance': -10, 'stop': -100, 'reverse': -2, 'successorScore': 100, 'distanceToFood': -1, 'fleeEnemy': -105.0}
  
class DefensiveReflexAgent(ReflexCaptureAgent):
  """
  A reflex agent that keeps its side Pacman-free.
  """
  
  #The 'memory' of the agent, to keep track/status of farthest food.
  food1 = None

  def chooseAction(self, gameState):
    """
    Overrided to have a unique condition for the DefensiveReflexAgent 
    (camp closest food to enemy).
    """
    actions = gameState.getLegalActions(self.index)
    

    #Sorting will help determine closest food to enemy.
    foodLeft = self.getFoodYouAreDefending(gameState).asList()
    foodLeft.sort()

    #Overrides default defensiveReflexAgentBehavior if no enemy is spotted.
    enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
    invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
    if(len(invaders) == 0):
      
      #If team is red, reverse list to get closest to enemy (closest to blue is a higher x val)
      if gameState.isOnRedTeam(self.index):
        foodLeft.reverse()
    
      #Starting route (to farthest food)
      if DefensiveReflexAgent.food1 == None:
        DefensiveReflexAgent.food1 = foodLeft[0]
      
      #If food farthest food is eaten, travel to the closest food from the farthest food.
      #after travelling to where the farthest food was.
      elif not gameState.hasFood(DefensiveReflexAgent.food1[0], DefensiveReflexAgent.food1[1]):
        myPos = gameState.getAgentPosition(self.index)
        if myPos[0] == DefensiveReflexAgent.food1[0] and myPos[1] == DefensiveReflexAgent.food1[1]:
          bestDist = 9999
          new_food = None
          for food in foodLeft:
            dist = self.getMazeDistance(food,DefensiveReflexAgent.food1)
            if dist < bestDist:
              new_food = food
              bestDist = dist
        DefensiveReflexAgent.food1 = new_food
      
      #Finds moveable action to lead to food location.
      bestDist = 9999
      for action in actions:
        successor = self.getSuccessor(gameState, action)
        pos2 = successor.getAgentPosition(self.index)
        dist = self.getMazeDistance(pos2,DefensiveReflexAgent.food1)
        if dist < bestDist:
          bestAction = action
          bestDist = dist
      return bestAction
    
    #Behavior of copied DefensiveReflexiveAgent.
    else:
      values = [self.evaluate(gameState, a) for a in actions]

      maxValue = max(values)
      bestActions = [a for a, v in zip(actions, values) if v == maxValue]

      if len(foodLeft) <= 2 or gameState.getAgentState(self.index).numCarrying > 2:
        bestDist = 9999
        for action in actions:
          successor = self.getSuccessor(gameState, action)
          pos2 = successor.getAgentPosition(self.index)
          dist = self.getMazeDistance(self.start,pos2)
          if dist < bestDist:
            bestAction = action
            bestDist = dist
        return bestAction

    return random.choice(bestActions)

  
  def getFeatures(self, gameState, action):
    """
    Helps modify the priority of a decision based on
    its current state.
    """
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)

    myState = successor.getAgentState(self.index)
    myPos = myState.getPosition()

    # Computes whether we're on defense (1) or offense (0)
    features['onDefense'] = 1
    if myState.isPacman: features['onDefense'] = 0

    # Computes distance to invaders we can see
    enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
    invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
    features['numInvaders'] = len(invaders)
    if len(invaders) > 0:
      dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
      features['invaderDistance'] = min(dists)

    if action == Directions.STOP: features['stop'] = 1
    rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
    if action == rev: features['reverse'] = 1

    return features

  def getWeights(self, gameState, action):
    """
    Default weight used to determine a move.
    Higher value means a higher priority.
    """
    return {'numInvaders': -1000, 'onDefense': 100, 'invaderDistance': -10, 'stop': -100, 'reverse': -2}