from classes import Graph, Edge, Vertex
import matplotlib.pyplot as plt

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
    print 'start box', im[rowi][coli] # should be nonzero now
    print 'start index', rowi, coli
    return [currentrow, currentcol]

# return coordinates of next pixel to go to in neighborhood
def check_neighborhood(im, nrow, ncol, currentrow, currentcol, r, prev_visited):
    # figure out range of indices to check in neighborhood
    row_start = max(currentrow-r, 0)
    row_end = min(currentrow+r, nrow-1)
    col_start = max(currentcol-r, 0)
    col_end = min(currentcol+r, ncol-1)

    # find index of neighbor with highest sum RGB
    row_min, col_min = None, None
    max_val = 0
    for rowi in range(row_start, row_end+1):
        for coli in range(col_start, col_end+1):
            if not (rowi == currentrow and coli == currentcol):
                if (rowi, coli) not in prev_visited:
                    print rowi, coli, im[rowi][coli]
                    rowsum = sum(im[rowi][coli])
                    if rowsum >= max_val:
                        max_val = rowsum
                        row_min = rowi
                        col_min = coli
    print 'maxval', max_val, row_min, col_min
    prev_visited.add((row_min, col_min))
    return row_min, col_min

def main():
    graph = Graph()
    tests()
    char_index = 11
    for person_index in range(1, 2):
        filename = "lao_images/000" + str(person_index) + "_" + str(char_index) + ".bmp"
        
    # read image as 2D numpy array
    im = plt.imread(filename)
    nrow, ncol = im.shape[0], im.shape[1]
    # find starting pixel (first nonzero pixel)
    currentrow, currentcol = find_start(im, nrow, ncol)

    r = 1 # radius around current pixel to be checked
    prev_visited = set()
    prev_visited.add((currentrow, currentcol))
    numiter = 10
    for i in range(numiter):
        currentrow, currentcol = check_neighborhood(im, nrow, ncol, currentrow, \
                                                        currentcol, r, prev_visited)
        print prev_visited


if __name__ == "__main__":
    main()
