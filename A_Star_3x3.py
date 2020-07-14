# imports the defined packages
import copy
import operator

# creates a current and goal list to iterate through
# goal = [[1, 2, 3], [4, 5, 6], [7, 8, '*']]
# current = [[1, 8, 6], [4, 3, 7], [5, 2, '*']]
goal = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, '*']]
current = [[3, 2, 4, 7], [13, 12, 1, 8], [10, 5, 11, 6], [15, 14, 9, '*']]


# creates a class with the respective elements and functions
class Node:
    # init/main function with respective params
    def __init__(self, graph, cost, moveP, moveDir, parentPath=None, sol=None):
        self.moveDir = moveDir  # the direction
        if sol is None:
            sol = goal  # the goal list
        self.graph = moveP(copy.deepcopy(graph))  # deep copies the current graph
        self.cost = cost + 1  # cost + 1 or g(n), number of iterations
        self.f = self.cost + manDistance(self.graph, sol)  # f(n), this is used to sort the list
        self.h = manDistance(self.graph, sol)  # h(n) is assigned the manhattan distance
        # if the parentPath is not None and not first iteration
        if parentPath is not None and parentPath != -1:
            self.path = copy.deepcopy(parentPath)
            self.path.append(graph)
        # elif the parent path is first iteration
        elif parentPath == -1:
            self.path = []
        # else append the graph and math path empty in such situation
        else:
            self.path = []
            self.path.append(graph)


# gets the position of the '*' in the graph
def getPos(graph):
    for row in range(len(graph)):
        for column in range(len(graph[row])):
            if graph[row][column] == '*':
                return row, column


# returns a graph, null function, for first position
def returnGraph(graph):
    return graph


# function to calculate manhattan distance
def manDistance(graph, answer):
    disSum = 0
    # for row and column in current list
    for row in range(len(graph)):
        for column in range(len(graph[row])):
            predict = ()
            lookNum = graph[row][column]
            actual = (row, column)
            # if the look variable is not the star
            if lookNum != '*':
                # for row and column in goal list
                for rGoal in range(len(answer)):
                    for cGoal in range(len(answer[rGoal])):
                        if answer[rGoal][cGoal] == lookNum:
                            predict = (rGoal, cGoal)
                # get the absolute value of both the row1 - row2 and column1 - column2 values, add it to disSum
                disSum += abs(actual[0] - predict[0]) + abs(actual[1] - predict[1])
    return disSum


# function for move left
def moveLeft(graph):
    # call the getPos function to find position
    posStar = getPos(graph)
    # subtract 1 from the value to move it to left
    if posStar[1] > 0:
        graph[posStar[0]][posStar[1]], graph[posStar[0]][posStar[1] - 1] = graph[posStar[0]][posStar[1] - 1], '*'
    return graph


# function for move right
def moveRight(graph):
    # call the getPos function to find position
    posStar = getPos(graph)
    # add 1 to the value to move it right
    if posStar[1] < len(graph) - 1:
        graph[posStar[0]][posStar[1]], graph[posStar[0]][posStar[1] + 1] = graph[posStar[0]][posStar[1] + 1], '*'
    return graph


# function for move up
def moveUp(graph):
    # call the getPos function to find position
    posStar = getPos(graph)
    # subtract 1 from the value to move it up
    if posStar[0] > 0:
        graph[posStar[0]][posStar[1]], graph[posStar[0] - 1][posStar[1]] = graph[posStar[0] - 1][posStar[1]], '*'
    return graph


# function for move down
def moveDown(graph):
    # call the getPos function to find position
    posStar = getPos(graph)
    # add 1 to the value to move it down
    if posStar[0] < len(graph) - 1:
        graph[posStar[0]][posStar[1]], graph[posStar[0] + 1][posStar[1]] = graph[posStar[0] + 1][posStar[1]], '*'
    return graph


# the main function
def main():
    # declare the specified variables
    fPos = Node(current, -1, returnGraph, -1, goal)
    currentNode = fPos
    frontierQ = []
    visitedNodes = []
    moveList = [moveLeft, moveRight, moveUp, moveDown]
    strMove = ['left', 'right', 'up', 'down']
    currentPath = []

    # use this loop to iterate through every current nodes if the h(n) is greater than 0
    while currentNode.h > 0:
        print(currentNode.graph)
        # for all directional values- left, right, up, and down
        for direction in moveList:
            index = moveList.index(direction)
            moved = direction(copy.deepcopy(currentNode.graph))
            # if the same as previous and not in visited nodes then only proceed
            if currentNode.graph != moved and (moved not in visitedNodes):
                currentNode.moveDir = strMove[index]
                frontierQ.append(
                    Node(currentNode.graph, currentNode.cost, direction, currentNode.moveDir, currentNode.path))
        # sort the function based on 'f' value
        sortedQ = sorted(frontierQ, key=operator.attrgetter('f'), reverse=True)
        # remove the list value from sortedQ and frontierQ
        currentNode = sortedQ.pop()
        frontierQ.remove(currentNode)
        # append the current node directional string and graph to respective lists
        currentPath.append(currentNode.moveDir)
        visitedNodes.append(currentNode.graph)

    # print the items in the specified way
    [print(*item, sep=', ') for item in fPos.graph]
    print('\n')
    [print(*item, sep=', ') for item in currentNode.graph]
    print('\n')
    print('FINISHED\n')
    print(f'Moves taken: {currentNode.cost} steps')


# calls the main function
main()

