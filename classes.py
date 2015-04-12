import math
import matplotlib.pyplot as plt 
import numpy as np
import __builtin__

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
                    
        print "Dim of matrix after " + str(len(self.adjmatrix)) + " " + str(len(self.adjmatrix[0]))
        
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

    # return vertex with averaged x and y coordinates
    def avgVertex(vertex1, vertex2):
        meanx = vertex1.x + (vertex2.x - vertex1.x)/2
        meany = vertex1.y + (vertex2.y - vertex1.y)/2
        newv = Vertex(meanx, meany)
        return newv
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

        
    
