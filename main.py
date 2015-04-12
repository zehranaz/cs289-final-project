from classes import Graph, Edge, Vertex

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
    graph.print_adjmatrix()
    graph.print_vertexlst()


def main():
    """ for Tamil characters dataset
    # convert tiff file to binary array
    im = plt.imread(tiff_file) # converts to type numpy.ndarray
    nrow = im.shape[0]
    ncol = im.shape[1]
    binim = np.zeros((nrow, ncol)) # 2D binary array
    for row in range(nrow):
        for col in range(ncol):
            # 1 if black, 0 if white
            binim[row][col] = 1 if 0 in im[row][col] else 0
    """
    graph = Graph()
    tests()

    # thin binary array to unit thickness

    # walk through every row in 2D array until see something black

if __name__ == "__main__":
    main()
