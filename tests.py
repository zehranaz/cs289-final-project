from classes import Graph, Edge, Vertex, MatchPoints, findPaths, findAllPaths, CrossOver
from random import randint
from crossover import read_graph_from_file
from bmp_to_graphs import produce_graphs

def output_graph(graph):
    vstr = ""
    vertices = graph.getVertexes()
    for i, vertex in enumerate(vertices):
        vstr += "(" + str(vertex.get_x()) + "," + str(vertex.get_y()) + ")"
        if i != len(vertices)-1:
            vstr += ","
    print vstr
    adjMatrix = graph.getAdjMatrix()
    res = ""
    for j, row in enumerate(adjMatrix):
        res += "{"
        for i, item in enumerate(row):
            if item == None:
                res += str(0)
            else:
                res += str(1)
            if i != len(row)-1:
                res += ","
        res += "}"
        if j != len(adjMatrix)-1:
            res += ","
    print res

def output_graphs():
    chars = [11]
    persons = range(1,5)
    num_persons = len(persons)
    graphs = produce_graphs(chars, persons, "coords")
    for char in chars:
        for i in range(num_persons-1):
            graph = graphs[char][persons[i]-1]
            output_graph(graph)

def test_read_crossovers():
    chars = [11]
    persons = range(1,8)
    for char in chars:
        for person in persons:
            graph = read_graph_from_file("000" + str(person) + "_" + str(char) + ".pkl")
            graph.print_graph()

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
    graph.addVertex(v2)
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
        g1 = makeTestGraph(45)
    if g2 == None:  
        g2 = makeTestGraph(35)
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
    g1 = makeTestGraph(35)
    print "g1 numvertices = ", len(g1.getVertexes())
    g1.print_vertexlst()
    g2 = makeTestGraph(25)
    print "g2 numvertices = ", len(g2.getVertexes())
    g2.print_vertexlst()
    # Since this makes the same graph, CrossOver should return same thing
    
    newGraph = CrossOver(g1, g2)
    print "New Graph = ", len(newGraph.getVertexes())
    newGraph.print_vertexlst()
    print "end New Graph"

# Extensive testing of CrossOver generation
def testCrossovers():
    for i in range(50):
        print "This is test ", i
        testCrossover()

def graph_readings_tests():
    #graph = makeSpecificGraph()
    #save_graph_to_file(graph, "test_graph")
    graph = read_graph_from_file("pkl_files/test_graph.pkl")
    graph.print_graph()

def main():
    #testMatch()
    testCrossover()
    #testRemoveVertex()
    #testFindPaths()
    #testGetNeighbors()

if __name__ == "__main__":
    main()
