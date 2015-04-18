from classes import Graph, Edge, Vertex, MatchPoints
from random import randint

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

def makeTestGraph():
    graph = Graph()
    # Pixels always between 33 x 48
    for i in range(5):
        v = Vertex(int(randint(0,33)), int(randint(0,48)))
        graph.addVertex(v)
    return graph


def testMatch():
    g1 = makeTestGraph()
    g2 = makeTestGraph()
    print "Graph 1:"
    g1.print_vertexlst()

    print "Graph 2:"
    g2.print_vertexlst()
    
    matches = MatchPoints(g1, g2, threshold = 4)
    print matches
    for v1,v2 in matches:
        print "(", v1.print_out(), v2.print_out(), ")",

def main():
    graph = Graph()
    #tests()
    testMatch()

    char_index = 11
    for person_index in range(1, 2):
        filename = "lao_images/000" + str(person_index) + "_" + str(char_index) + ".bmp"
        
    # walk through every row in 2D array until see something black


if __name__ == "__main__":
    main()
