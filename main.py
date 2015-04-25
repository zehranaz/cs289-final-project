#!/usr/bin/env python
from classes import Graph, Edge, Vertex, MatchPoints, GenerateNewLetter, findPaths
from random import randint
import matplotlib.pyplot as plt
import numpy as np
<<<<<<< HEAD
import cv2
import thinning
import coords_to_img

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

def testFindPaths():
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

    path = []
    findPaths(v2, v3, 1, graph, path)
    for v in path:
        print v.print_out()


def testMating():
    g1 = makeTestGraph()
    g2 = makeTestGraph()
    matches = testMatch(g1, g2)
    newGraph = GenerateNewLetter(g1, g2, matches)
    newGraph.print_vertexlst()


def main() :
    testFindPaths()


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

def neighbors_of_vector(vec, currentrow, currentcol):
    # horizontal vector
    if vec == [0, 1] or vec == [0, -1]:
        return [(currentrow+1, currentcol), (currentrow-1, currentcol)]
    # vertical vector
    if vec == [1, 0] or vec == [-1, 0]:
        return [(currentrow, currentcol-1), (currentrow, currentcol+1)]
    # downwards diagonal vector
    if vec == [1, 1] or vec == [-1, -1]:
        return [(currentrow-1, currentcol+1), (currentrow+1, currentcol-1)]
    # upwards diagonal vector
    if vec == [-1, 1] or vec == [1, -1]:
        return [(currentrow+1, currentcol+1), (currentrow-1, currentcol-1)]


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
                        '''
                        curr_vector = [rowi - currentrow, coli - currentcol]
                        angle = angle_between(prev_vector, curr_vector)

                        if angle < min_angle:
                            min_angle = angle
                            row_min = rowi
                            col_min = coli
                        '''
                        prev_visited.add((rowi, coli))
                        pts_to_traverse.append((rowi, coli))
                        # check_neighborhood(im, nrow, ncol, rowi, coli, r, prev_vector, prev_visited)
    
    '''
    print 'min_angle', min_angle, row_min, col_min
    prev_visited.add((row_min, col_min))
    start_new_vertex = min_angle > 0.1
    '''
    # add left and right neighbors to visited
    '''
    if not start_new_vertex:
        for neighbor in neighbors_of_vector(curr_vector, currentrow, currentcol):
            prev_visited.add(neighbor)
    '''
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
    sum_of_distances = 0.
    for (v1, v2) in matching_vertices:
        sum_of_distances += v1.EuclidDist(v2)
    return sum_of_distances

def build_graph(im):
    nrow, ncol = im.shape[0], im.shape[1]

    # for keeping track of visited
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

    plt.axis("off")
    plt.imshow(imdup)
    plt.show()

    # graph.print_graph()

    return graph

def main_victoria():

    graphs = []
    char_indices = [11, 12]
    person_indices = range(1, 6)
    for char_index in char_indices:
        for person_index in person_indices:
            # produce bmp
            coords_to_img.convert_images(person_indices, char_indices)

            # thinning
            infile = "lao_images/000" + str(person_index) + "_" + str(char_index) + ".bmp"
            outfile = "lao_images/000" + str(person_index) + "_" + str(char_index) + "_thin.bmp"
            thinning.thin_image(infile, outfile)

            # build graph
            im = plt.imread(outfile)
            graphs.append(build_graph(im))

    print fitness_between_nodes(graphs[2], graphs[3], 50)


    '''
    prev_row = currentrow
    prev_col = currentcol
    prev_vector = [1,0] # unit vector for previous direction
    '''
    '''
    while True:
        print 'prev_visited', prev_visited
        print 'checking', currentrow, currentcol

        if currentrow == None:
            break
        # mark the visited pixels in image
        imdup[currentrow][currentcol] = 1
        
        if start_new_vertex:
            new_vertex = Vertex(currentrow, currentcol)
            # create an edge between this and previous vertex
            dist = new_vertex.EuclidDist(prev_vertex)
            e = Edge(prev_vertex, new_vertex, dist, False)
            graph.addVertex(new_vertex)
            graph.addEdge(e)
            prev_vertex = new_vertex
        prev_vector = [currentrow - prev_row, currentcol - prev_col]
        prev_row = currentrow
        prev_col = currentcol
    '''

    

    # make the last point a vertex and add it to graph
    '''
    new_vertex = Vertex(prev_row, prev_col)
    dist = new_vertex.EuclidDist(prev_vertex)
    e = Edge(prev_vertex, new_vertex, dist, False)
    graph.addVertex(new_vertex)
    graph.addEdge(e)
    '''


if __name__ == "__main__":
    main_victoria()
