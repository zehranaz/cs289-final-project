# Representation for the hand writing problem

import math
import matplotlib.pyplot as plt 
import numpy as np

class Vertex:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.visited = False
    def EuclidDist(self, vertex):
        return math.sqrt(pow(self.x - vertex.x, 2) + pow(self.y - vertex.y, 2))
    def printout(self):
    	print str(self.x) + ", " + str(self.y)
        
class Edge:
    def __init__(self, startv, endv, maxv, iscurve):
        self.starv = startv
        self.endv = endv
        self.maxv = maxv
        self.iscurve = iscurve

class Graph:
    def __init__(self):
        vertexlst = []
        adjmatrix = [] # will be a nested list of form [[1, 0], [0, 1]], also a symmetric matrix
        
    def addVertex(self, vertex):
        vertexlst.append(vertex)
        for row in adjmatrix:
            row.append(0) # add a new column
        adjmatrix.append([0 * len(vertexlst)]) # add a new row (there's probably a more efficient way)
        
    def addEdge(self, edge, vertexind1, vertexind2):
        adjmatrix[vertexind1][vertexind2] = edge # assuming this is pass by reference
    
    # return vertex with averaged x and y coordinates
    def avgVertex(vertex1, vertex2):
        meanx = vertex1.x + (vertex2.x - vertex1.x)/2
        meany = vertex1.y + (vertex2.y - vertex1.y)/2
        newv = Vertex(meanx, meany)
        return newv
        
def tests():
    v1 = Vertex(2,3)
    v1.printout()
    v2 = Vertex(0,0)
    dist = v1.EuclidDist(v2)
    e = Edge(v1, v2, dist, False)
    graph = Graph()
    graph.addVertex(v1)
    print(graph) # TODO: may want to make print function for Graph
    graph.addVertex(v1)
    print(graph)
    graph.addEdge(e)
    print(graph)
    
    
        
def main():
    # im = plt.imread(tiff_file) # converts to type numpy.ndarray
    # nrow = im.shape[0]
    # ncol = im.shape[1]
    # binim = np.zeros((nrow, ncol)) # 2D binary array
    # for row in range(nrow):
    #     for col in range(ncol):
    #         binim[row][col] = 1 if 0 in im[row][col] else 0
    tests()


if __name__ == "__main__":
    main()
