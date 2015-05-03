#!/usr/bin/env python
from classes import Graph, Edge, Vertex, MatchPoints, findPaths, findAllPaths, CrossOver
from random import randint
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import thinning
import coords_to_img
import os.path
import copy
import sys
import math
import copy

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    angle = np.arccos(np.dot(v1_u, v2_u))
    if np.isnan(angle):
        if (v1_u == v2_u).all():
            return 0.0
        else:
            return np.pi
    return (angle / np.pi) * 180.

def tests():
    v1 = Vertex(2,3)
    v2 = Vertex(0,0)
    v1.print_out()
    v2.print_out()
    print "\n"
    
    dist = v1.EuclidDist(v2)
    e = Edge(v1, v2, dist, False)

    graph = Graph()

    graph.addVertex(v1)
    # graph.print_adjmatrix()
    # graph.print_vertexlst()

    graph.addVertex(v2)
    # graph.print_adjmatrix()
    # graph.print_vertexlst()

    graph.addEdge(e)
    graph.print_adjmatrix()
    graph.print_vertexlst()

    v3 = Vertex(4,5)
    graph.addVertex(v3)
    e2 = Edge(v1, v3, v3.EuclidDist(v1), True)
    graph.addEdge(e2)
    print("Real Print starts here ")
    graph.print_graph()

def makeSpecificGraph():
    # Set up graph
    graph = Graph()

    v1 = Vertex(2,3)
    graph.addVertex(v1)

    v2 = Vertex(0,0)
    graph.addVertex(v2)
    
    dist = v1.EuclidDist(v2)
    e = Edge(v1, v2, dist, False)
    graph.addEdge(e)

    v3 = Vertex(4,5)
    graph.addVertex(v3)
    
    e2 = Edge(v1, v3, v3.EuclidDist(v1), True)
    graph.addEdge(e2)

    v4 = Vertex(10,20)
    graph.addVertex(v4)

    e3 = Edge(v3, v4, v3.EuclidDist(v4), False)
    graph.addEdge(e3)
    return graph

def testGetNeighbors():
    graph = makeSpecificGraph()
    vertexes = graph.getVertexes()
    for index in range(len(vertexes)):
        print "For vertex " + str(index)
        neighbors = graph.getNeighborVertexes(vertexes[index])

        for vertex in neighbors:
            print vertex.print_out()

# returns graph with numPoints vertexes; random number of vertexes if None specified
def makeTestGraph(numPoints=None):
    graph = Graph()
    # Pixels always between 33 x 48
    if not numPoints:
        numPoints = randint(5,10)
    for i in range(numPoints):
        v = Vertex(int(randint(0,33)), int(randint(0,48)))
        graph.addVertex(v)
    return graph

def testMatch(g1=None, g2=None):
    if g1 == None:
        g1 = makeTestGraph()
    if g2 == None:  
        g2 = makeTestGraph()
    print "Graph 1:"
    g1.print_vertexlst()

    print "Graph 2:"
    g2.print_vertexlst()
    
    matches = MatchPoints(g1, g2, threshold = 4)
    print matches
    for v1,v2 in matches:
        print "(", v1.print_out(), v2.print_out(), ")",
    return matches

def testRemoveVertex():
    graph = makeSpecificGraph()
    graph2 = copy.deepcopy(graph)
    print "Before: "
    graph.print_graph()    
    vertexToRemove = graph.getVertexes() [1]
    graph.removeVertex(vertexToRemove)
    print "After: "
    graph.print_graph()
    print "Copyed Graph "
    graph2.print_graph()

def testCopy():
    graph

def testFindPaths():
    # Set up graph
    graph = makeSpecificGraph()
    vertexes = graph.getVertexes()
    v1 = vertexes[0]
    v2 = vertexes[1]
    v3 = vertexes[2]
    v4 = vertexes[3]

    path = []
    findPaths(v2, v3, 1, graph, path)
    for v in path:
        print v.print_out()

    print "New one: "
    path = []
    findPaths(v2, v4, 2, graph, path)
    for v in path:
        print v.print_out()

    allPaths = findAllPaths(graph)
    # print allPaths
    # index = 0
    # for path in allPaths:

    #     print "Path " + str(index)
    #     index += 1
    #     # print "start: "
    #     # print path[0].print_out()
    #     # print "end: " 
    #     # print  path[len(path) - 1].print_out()
    #     print "Path list: "
    #     for v in path:
    #         print v.print_out(), 
    #     #print allPaths[path]

def testCrossover():
    g1 = makeSpecificGraph()
    g2 = makeSpecificGraph()
    # Since this makes the same graph, CrossOver should return same thing
    newGraph = CrossOver(g1, g2)
    newGraph.print_graph()

    print "Old Graph = "
    g1.print_graph()

def main():
    testCrossover()
    #testRemoveVertex()
    #testFindPaths()
    #testGetNeighbors()

# walk through image diagonally until see something black
def find_start(im, nrow, ncol):
    rowi, coli, i = 0, 0, 0
    while rowi < nrow and coli < ncol and im[rowi][coli].all() == 0:
        if coli < i:
            rowi -= 1
            coli += 1
        elif coli == i:
            i += 1
            rowi = i
            coli = 0
    currentrow = rowi
    currentcol = coli
    # print 'start box', im[rowi][coli] # should be nonzero now
    # print 'start index', rowi, coli
    return [currentrow, currentcol]

# return coordinates of next pixel to go to in neighborhood
def check_neighborhood(im, nrow, ncol, currentrow, currentcol, r, prev_visited):
    # figure out range of indices to check in neighborhood
    row_start = max(currentrow-r, 0)
    row_end = min(currentrow+r, nrow-1)
    col_start = max(currentcol-r, 0)
    col_end = min(currentcol+r, ncol-1)

    # find nonzero neighbor with least deviation from previous direction
    row_min, col_min = None, None
    min_angle = 360.
    pts_to_traverse = []
    for rowi in range(row_start, row_end+1):
        for coli in range(col_start, col_end+1):
            # if not the current pixel
            if not (rowi == currentrow and coli == currentcol):
                # if not a previously visited pixel
                if (rowi, coli) not in prev_visited:
                    # if not black
                    if not im[rowi][coli] == 0:
                        prev_visited.add((rowi, coli))
                        pts_to_traverse.append((rowi, coli))

    return pts_to_traverse, prev_visited # row_min, col_min, start_new_vertex

def traverse(im, graph, nrow, ncol, currentrow, currentcol, r, imdup, prev_vertex, prev_visited):
    # get children
    pts_to_traverse, prev_visited = check_neighborhood(im, nrow, ncol, currentrow, \
                                                    currentcol, r, prev_visited)
    # if it's a point of interseciton, add to graph
    if len(pts_to_traverse) > 1:
        imdup[currentrow][currentcol] = 1
        new_vertex = Vertex(currentrow, currentcol)
        # if not already in the graph, add vertex and edge to previous vertex
        if not graph.has_vertex(new_vertex):
            # create an edge between this and previous vertex
            dist = new_vertex.EuclidDist(prev_vertex)
            e = Edge(prev_vertex, new_vertex, dist, False)
            graph.addVertex(new_vertex)
            graph.addEdge(e)
            prev_vertex = new_vertex

    # explore children
    for pt in pts_to_traverse:
        traverse(im, graph, nrow, ncol, pt[0], pt[1], r, imdup, prev_vertex, prev_visited)

    return imdup

def fitness_between_nodes(g1, g2, threshold):
    matching_vertices = MatchPoints(g1, g2, threshold)
    # if insufficient number of points matched
    print 'number of matching nodes', len(matching_vertices), g1.numVertices()
    if len(matching_vertices) < g1.numVertices():
        print 'too few matches'
        return sys.maxint
    sum_of_distances = 0.
    for (v1, v2) in matching_vertices:
        sum_of_distances += v1.EuclidDist(v2)
    return math.sqrt(sum_of_distances)

def build_graph(im, fbasename):
    # get image dimensions
    nrow, ncol = im.shape[0], im.shape[1]

    # for keeping visual of points added to graph
    imdup = np.zeros(im.shape)

    # initiate graph
    graph = Graph()

    # find starting pixel (first nonzero pixel)
    currentrow, currentcol = find_start(im, nrow, ncol)
    prev_vertex = Vertex(currentrow, currentcol)
    graph.addVertex(prev_vertex)

    r = 1 # radius around current pixel to be checked
    prev_visited = set()
    prev_visited.add((currentrow, currentcol))

    # traverse through image and build graph
    imdup = traverse(im, graph, nrow, ncol, currentrow, currentcol, r, imdup, prev_vertex, prev_visited)

    # save to file
    fig1 = plt.gcf()
    plt.imshow(imdup, cmap='Greys')
    plt.draw()
    fig1.savefig(fbasename + "_graph.jpg")
    
    return graph

# order of work:
# 1) produce bmp from coordinates
# 2) thin the bmp's
# 3) build the graph from the thinned image
def produce_graphs(char_indices, person_indices, jobtype):
    redo_thinning = False
    graphs = defaultdict(list)
    for char_index in char_indices:
        for person_index in person_indices:
            if person_index < 10:
                fbasename = "lao_images/000" + str(person_index) + "_" + str(char_index)
            else:
                fbasename = "lao_images/00" + str(person_index) + "_" + str(char_index)
            if jobtype == "coords":
                # produce bmp from list of coordinates if it doesn't already exist
                if not os.path.isfile(fbasename + ".bmp"):
                    coords_to_img.convert_images(person_indices, char_indices)
                if not os.path.isfile(fbasename + "_thin.bmp"):
                    redo_thinning = True
            elif jobtype == "thin":
                infile = fbasename + ".bmp"
                outfile = fbasename + "_thin.bmp"
                thinning.thin_image(infile, outfile)
            elif jobtype == "graph":
                im = plt.imread(fbasename + "_thin.bmp")
                graphs[char_index].append(build_graph(im, fbasename))
    if jobtype == "graph":
        return graphs
    elif redo_thinning:
        return produce_graphs(char_indices, person_indices, "thin")
    else:
        return produce_graphs(char_indices, person_indices, "graph")


def main_victoria():
    char_indices = [11, 12, 9, 5, 18]
    person_indices = range(1, 10)
    
    # produce graphs for each character
    graphs = produce_graphs(char_indices, person_indices, "coords")

    # evaluate fitness between one graph and all other graphs
    test_char_indices = [11, 12, 9, 5, 18]
    test_person_indices = range(0, 9)
    correct_classifications = defaultdict(int)

    for test_char_index in test_char_indices:
        for test_person_index in test_person_indices:
            test_graph = graphs[test_char_index][test_person_index]
            closest_char = None
            min_fitness_val = sys.maxint
            print 'classifying character', test_char_index, 'by person', test_person_index
            for char_index in char_indices:
                for person_index in person_indices:
                    if not (char_index == test_char_index and person_index == test_person_index):
                        g1 = test_graph
                        g2 = graphs[char_index][person_index-1]
                        fitness = fitness_between_nodes(g1, g2, sys.maxint)
                        if fitness < min_fitness_val:
                            min_fitness_val = fitness
                            closest_char = char_index
                        print person_index, char_index, fitness
            print 'classification of character', test_char_index, 'by person', \
                    test_person_index, 'is', closest_char, 'with fitness', min_fitness_val
            if closest_char == test_char_index:
                correct_classifications[test_char_index] += 1
    for char,num_correct in correct_classifications.items():
        print char, num_correct, num_correct / float(len(test_person_indices))


if __name__ == "__main__":
    main_victoria()
