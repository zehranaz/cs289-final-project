import math
import numpy as np
import __builtin__
import sys
import copy
from random import randint, random 

class Vertex:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.visited = False
    def EuclidDist(self, vertex):
        return math.sqrt(pow(self.x - vertex.x, 2) + pow(self.y - vertex.y, 2))
    def get_x(self):
        return self.x
    def get_y(self):
        return self.y
    def change_x(self, new_x):
        self.x = new_x
    def change_y(self, new_y):
        self.y = new_y
    def changeVertex(self, newVertex):
        self.change_x(newVertex.x)
        self.change_y(newVertex.y)
    def print_out(self):
        return str(self.x) + ", " + str(self.y)
    # returns coordinates of self averaged with vertex2
    def avgVertex(self, vertex2):
        meanx = self.x + (vertex2.x - self.x)/2
        meany = self.y + (vertex2.y - self.y)/2
        newv = Vertex(meanx, meany)
        return newv
    def __eq__(self, other):
        if other == None:
            return False
        if self.x == other.x and self.y == other.y:
            return True
        return False

class Edge:
    def __init__(self, startv, endv, max_dist, iscurve):
        self.startv = startv
        self.endv = endv
        self.max_dist = max_dist
        self.iscurve = iscurve
   
    def get_startv(self):
        return self.startv
    
    def set_startv(self, newStartV):
        self.startv = newStartV
    
    def get_endv(self):
        return self.endv
    
    def set_endv(self, newEndV):
        self.endv = newEndV
   
    def get_max_dist(self):
        return self.max_dist
   
    def print_out(self):
        return "Edge Info - Start: " + self.startv.print_out() + ", End: " + self.endv.print_out() + ", Max Dist: " + str(self.max_dist) + ", is Curve: " + str(self.iscurve)

class Graph:
    def __init__(self):
        self.vertexlst = []
        self.adjmatrix = [] # will be a nested list of form [[1, 0], [0, 1]], also a symmetric matrix

    def has_vertex(self, vertex):
        for v in self.vertexlst:
            if v == vertex:
                return True
        return False

    def numVertices(self):
        return len(self.vertexlst)
        
    def getVertexes (self):
        return self.vertexlst

    def getAdjMatrix (self):
        return self.adjmatrix

    # Returns the edge between v1 and v2
    def getEdge(self, v1, v2):
        return self.adjmatrix[self.getIndexOfVertex(v1)][self.getIndexOfVertex(v2)]

    # Given a vertex, get its edges
    def getEdges (self, vertex):
        edges = []
        
        vIndex = getIndexOfVertex(vertex)
        if not vIndex:
            return []

        # Traverse appropriate row in matrix to find all edges of this vertex
        for i in range(len(self.adjmatrix [vIndex])):
            edge = self.adjmatrix[vIndex][i]
            if edge:
                edges.append(edge)
        return edges

    def getIndexOfVertex(self, vertex):
        index = None
        try: 
            # Find index of vertex in list
            index = self.vertexlst.index(vertex)
        except:
            pass
        return index

    # Given a vertex, finds other vertexes it shares and edge with
    def getNeighborVertexes(self, vertex):
        neighbors = []

        vIndex = self.getIndexOfVertex(vertex)
        if vIndex == None:
            return []

        # Traverse row to find out the vertexes 'vertex' connects to
        for i in range(len(self.adjmatrix [vIndex])):
            edge = self.adjmatrix[vIndex][i]
            if edge:
                neighbors.append(self.vertexlst[i])

        return neighbors

    def addVertex(self, vertex):
        num_vertices = len(self.vertexlst)
        # print "Vert len before " + str(num_vertices)
        
        self.vertexlst.append(vertex)
        # print "Vert len after " + str(len(self.vertexlst))

        # adding initial vertex
        if num_vertices == 0:
            self.adjmatrix = [[None]]
        else: 
            for row in self.adjmatrix:
                row.append(None)
            new_empty_row = [None for x in range(len(self.vertexlst))]
            self.adjmatrix.append(new_empty_row)  # add a new row 
        # print "Dim of matrix after " + str(len(self.adjmatrix)) + " " + str(len(self.adjmatrix[0]))

    # Takes a vertex and removes it.
    def removeVertex(self, vertex):
        vIndex = self.getIndexOfVertex(vertex)
        # print "VINDEX = " + str(vIndex) + " and vertex = ",
        # print vertex.print_out()
        if not vIndex == None:
            self.adjmatrix = np.delete(self.adjmatrix, vIndex, 0)
            self.adjmatrix = np.delete(self.adjmatrix, vIndex, 1)
            self.vertexlst.remove(vertex)

    # find the index of each vertex in the vertex list and use it to locate it in the adj matrix   
    def addEdge(self, edge):
        start_ind = self.vertexlst.index(edge.get_startv())
        end_ind = self.vertexlst.index(edge.get_endv())

        # find index of start and end vertexes in vertex list, add edge both ways
        start_i = self.vertexlst.index(edge.get_startv())   
        end_i = self.vertexlst.index(edge.get_endv())
        self.adjmatrix[start_i][end_i] = edge # assuming this is pass by reference
        self.adjmatrix[end_i][start_i] = edge

    def print_adjmatrix(self):
        print "Adjacency Matrix"
        for row in self.adjmatrix:
            for col in row:
                if col == None:
                    print col
                else:
                    print col.print_out()
            print 
    def print_vertexlst(self):
        print "List of Vertexes"
        for elt in self.vertexlst:
            print elt.print_out()
        print 
    def print_graph (self):
        print "Graph"
        self.print_vertexlst()
        # self.print_adjmatrix()


# Given two graphs and threshold (float for the max dist acceptable between two matching points), find the corresponding points; return as list
# TODO: an improved version would be one that minimizes total distance between all nodes
def MatchPoints(g1, g2, threshold):
    ver1 = copy.deepcopy(g1.getVertexes())
    ver2 = copy.deepcopy(g2.getVertexes())

    matches = []

    # find two vertexes with min distance
    minDist = sys.maxint
    closestVertex = None
    while ver1 and ver2:
        # print "in while:", len(ver1), len(ver2)
        for v1 in ver1:
            # print "in for v1 in ver1:", len(ver1), len(ver2)
            if ver2:
                for v2 in ver2:
                    # print "in for v2 in ver2:", len(ver2)
                    dist = v1.EuclidDist(v2)
                    # print 'v1', v1.print_out(), 'v2', v2.print_out(), 'dist', dist
                    if dist < minDist:
                        minDist = dist
                        closestVertex = v2
                if closestVertex == None or minDist > threshold:
                    ver1.remove(v1)
                    # print "THERE IS AN UNMATCHED VERTEX! ", v1.print_out()
                elif closestVertex and minDist <= threshold:
                    matches.append((v1, closestVertex))
                    ver1.remove(v1)
                    # print "removed v1"
                    ver2.remove(closestVertex)  
            closestVertex = None
            minDist = sys.maxint
    return matches
        
# TODO: Needs optimization from dynamic programming storage (can store if there is a path in an adj matrix and lay out the paths there)
# Find path between a pair of vertexes i and j of len length    
def findPathsWithoutLast(iVertex, jVertex, length, graph, path):
    if length < 0:
        raise Exception("path length cannot be < zero")
    if length == 0:
        if jVertex in graph.getNeighborVertexes(iVertex):
            path.append(iVertex)
            return path
        else:
            return []
    else:
        # iterate through neighbors k of j and paths from i to k
        jNeighbors = graph.getNeighborVertexes(jVertex)
        for k in jNeighbors:
            pathFound = findPathsWithoutLast(iVertex, k, length-1, graph, path)
            # Do not add in k if already there
            if k in pathFound:
                path = []
                pathFound = []
                continue
            if not pathFound == []:
                 pathFound.append(k)
                 return pathFound
        return []

# Find between vertex i and j, then append j to the path list
def findPaths(iVertex, jVertex, length, graph, path):
    pathFound = findPathsWithoutLast(iVertex, jVertex, length, graph, path)
    if not pathFound == []:
        if jVertex not in pathFound: 
            pathFound.append(jVertex)
    # Check to make sure that the path length adds up (i.e. 'length' vertexes are between 'i' and 'j')
    if not len(pathFound) == length + 2:
        pathFound = []
    return pathFound


# Given a graph, finds all paths in it up to length_limit length
def findAllPaths(graph, length_limit=4):
    vertexList = graph.getVertexes()
    numVertexes = len(vertexList)
    pathsFound = []

    for length in range(length_limit + 1):
        for v1 in range(numVertexes-1):
            for v2 in range(v1 + 1, numVertexes):
                if v1 == v2:
                    continue
                paths = []
                paths = findPaths(vertexList[v1], vertexList[v2], length, graph, paths)
                if not paths == []:
                    pathsFound.append(paths)    # Save the paths
    return pathsFound

# Helper for finding matching paths. Takes vertex from g2 and finds the 
#  equivalent from g1 by using matches found
def findVertexInMatches(vertex, matches, vertexes2):
    index = vertexes2.index(vertex)    
    (vert1, _) = matches[index]
    return vert1

# Helpers for testing
def printVertexList(lst):
    for item in lst:
        print item.print_out(), " "
def printList(lst):
    for item in lst:
        print item

# Given graphs of letters, generates an "evolved" (averaged) graph
def CrossOver(g1, g2, threshold=5):

    print 'number of nodes', g1.numVertices(), g2.numVertices()
    # Generate edges
    paths1 = findAllPaths(g1)
    print "number of paths", len(paths1)
    
    # Make a graph with only the matched edges (to help us find all paths only with those vertexes)
    newGraph = copy.deepcopy(g2)    # to be returned
    graph2_copy = copy.deepcopy(g2) # for finding paths on matched vertexes
    
    matches = MatchPoints(g1, graph2_copy, threshold)
    print "number of matches: ", len(matches)

    # Pull out matched vertexes in graph 2
    v2matches = []
    for v1,v2 in matches:
        print v1.print_out(), 'to', v2.print_out()
        v2matches.append(v2)
    vertexes2 = newGraph.getVertexes()

    # Remove vertexes not matched
    to_remove = []
    for v2 in vertexes2:
        if v2 in v2matches:
            continue
        else:
            to_remove.append(v2) 

    for v in to_remove:
        graph2_copy.removeVertex(v)

    # Find paths in the newGraph
    paths2 = findAllPaths(graph2_copy)
    
    # just printing
    # print "paths1"
    # for p1 in paths1:
    #     for v1 in p1:
    #         print v1.print_out(), " ; ",
    #     print ""
    # print "paths2"
    # for p2 in paths2:
    #     for v2 in p2:
    #         print v2.print_out(), " ; ",
    #     print ""

    # Replace paths in newGraph
    for p2 in paths2:
        p1 = []
        for vertex2 in p2:
            # Find equivalent vertex1 thru matches
            if vertex2 not in v2matches:
                print "VERTEX NOT IN V2MATCHES...HOW DID IT GET HERE... ", vertex2.print_out()
                print "Matches are "
                printVertexList(v2matches)

                print "REAL matches list was: "
                for w1, w2 in matches:
                    print w1.print_out(), w2.print_out()
            vertex1 = findVertexInMatches(vertex2, matches, v2matches)
            # Append to matchPaths 
            if vertex1 == None:
                print "No Matching v1 found. THIS IS BAD. DEBUG this."
            p1.append(vertex1)

        # Find out whether p1 (or its reverse) exists in paths1
        if p1 in paths1 or p1.reverse() in paths1:
            # Replace common paths with 50% prob, based on how many common paths found
            print "in p1"
            if True: #random() < 1.:
                print "REPLACING A PATH"
                # edge for edge, add and delete
                #TODO: Might have to change i and i+1 around to fit if reverse path is found.
                for i in range(len(p2) - 1):
                    edge1 = g1.getEdge(p1[i], p1[i+1])
                    #TODO: might want to figure out why there is no edge in this path
                    if edge1:
                        edge1.set_startv(p2[i])
                        edge1.set_endv(p2[i+1])
                        newGraph.addEdge(edge1)
                    else:
                        print "MISSING EDGE in path...this is weird"

        p1 = []

    # update the matched vertexes in newGraph with their averages
    for v2 in vertexes2:
        if v2 in v2matches:
            # Get index into matches and change vertex to average of two matched v's
            print "AVERAGING VERTEX"
            v2Index = v2matches.index(v2)
            v2.changeVertex(matches[v2Index][0].avgVertex(v2))

    return newGraph

