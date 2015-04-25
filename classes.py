import math
import numpy as np
import __builtin__
import sys

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
    def print_out(self):
        return str(self.x) + ", " + str(self.y)
    # returns coordinates of self averaged with vertex2
    def avgVertex(self, vertex2):
        meanx = self.x + (vertex2.x - self.x)/2
        meany = self.y + (vertex2.y - self.y)/2
        newv = Vertex(meanx, meany)
        return newv

class Edge:
    def __init__(self, startv, endv, max_dist, iscurve):
        self.startv = startv
        self.endv = endv
        self.max_dist = max_dist
        self.iscurve = iscurve
    def get_startv(self):
        return self.startv
    def get_endv(self):
        return self.endv
    def get_max_dist(self):
        return self.max_dist
    def print_out(self):
        return "Edge Info - Start: " + self.startv.print_out() + ", End: " + self.endv.print_out() + ", Max Dist: " + str(self.max_dist) + ", is Curve: " + str(self.iscurve)

class Graph:
    def __init__(self):
        self.vertexlst = []
        self.adjmatrix = [] # will be a nested list of form [[1, 0], [0, 1]], also a symmetric matrix
        
    def getVertexes (self):
        return self.vertexlst

    def getAdjMatrix (self):
        return self.adjmatrix

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
        if not vIndex:
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
                    
        #print "Dim of matrix after " + str(len(self.adjmatrix)) + " " + str(len(self.adjmatrix[0]))
        
    # find the index of each vertex in the vertex list and use it to locate it in the adj matrix   
    def addEdge(self, edge):
        start_ind = self.vertexlst.index(edge.get_startv())
        end_ind = self.vertexlst.index(edge.get_endv())
        # print start_ind
        # print end_ind
        # print "Matrix dim are: "
        # print len(self.adjmatrix)
        # print len(self.adjmatrix[0])

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
        self.print_adjmatrix()


# Given two graphs and threshold (float for the max dist acceptable between two matching points), find the corresponding points; return as list
# TODO: an improved version would be one that minimizes total distance between all nodes
def MatchPoints(g1, g2, threshold):
    ver1 = g1.getVertexes()
    ver2 = g2.getVertexes()
    matches = []

    # find two vertexes with min distance
    minDist = sys.maxint
    closestVertex = None
    for v1 in ver1:
        for v2 in ver2:
            dist = v1.EuclidDist(v2)
            if dist < minDist:
                minDist = dist
                closestVertex = v2
        if not closestVertex == None and minDist <= threshold:
            matches.append((v1, closestVertex))
            ver2.remove(v2)
            ver1.remove(v1)
        closestVertex = None
        minDist = sys.maxint
    return matches

# Given graphs of letters and a list of matched points, generates an "evolved" (averaged) graph
def GenerateNewLetter(g1, g2, matches) :
    newGraph = Graph()
    # Generate average vertexes
    for v1, v2 in matches:
        newVertex = v1.avgVertex(v2)
        newGraph.addVertex(newVertex)
    # Generate average edges
    return newGraph
        
    
# TODO: Needs optimization from dynamic programming storage (can store if there is a path in an adj matrix and lay out the paths there)
# TODO: test this!!
# Find path between a pair of vertexes i and j of len length    
def findPaths(iVertex, jVertex, length, graph, path):
    if length == 0:
        if jVertex in graph.getNeighborVertexes(iVertex):
            path.append(iVertex)
            return path
        else:
            return []
    else:
        jNeighbors = graph.getNeighborVertexes(jVertex)
        for k in jNeighbors:
            pathFound = findPaths(iVertex, k, length-1, graph, path)
            if pathFound and not k in path:
                 pathFound.append(k)
                 return pathFound
        return []


def generateCrossover(g1, g2):
    matches = MatchPoints(g1, g2)
    adj1 = g1.getAdjMatrix()
    adj2 = g2.getAdjMatrix()
