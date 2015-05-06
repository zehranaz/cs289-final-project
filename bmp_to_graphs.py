from collections import defaultdict
import os.path
import numpy as np
import matplotlib.pyplot as plt
from classes import Graph, Vertex, Edge
import coords_to_img
import thinning

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
    

# Build Graph from image, save plot of graph to file, return graph
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
    # nested dictionary indexed by character index then person index
    graphs = defaultdict(lambda: defaultdict(list))
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
                graphs[char_index][person_index] = build_graph(im, fbasename)
    if jobtype == "graph":
        return graphs
    elif redo_thinning:
        return produce_graphs(char_indices, person_indices, "thin")
    else:
        return produce_graphs(char_indices, person_indices, "graph")