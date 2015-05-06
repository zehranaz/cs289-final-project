from classes import MatchPoints
from collections import defaultdict
import sys
import math

def fitness_between_nodes(g1, g2, threshold):
    matching_vertices = MatchPoints(g1, g2, threshold)
    # if insufficient number of points matched
    
    #print 'g1 num vertices = ', g1.numVertices(), 'and g2 num vertices = ', g2.numVertices()
    #print 'number of matching nodes', len(matching_vertices), 'out of ', max(g1.numVertices(), g2.numVertices())
    
    # if len(matching_vertices) < min(g1.numVertices(), g2.numVertices()):
    #     print 'too few matches'
    #     threshold = max(2*threshold, sys.maxint)
    #     return fitness_between_nodes(g1, g2, threshold)
    # sum over all distances between matched vertexes
    sum_of_distances = 1.
    max_vertices = max(g1.numVertices(), g2.numVertices())
    for (v1, v2) in matching_vertices:
        sum_of_distances += v1.EuclidDist(v2)
    # if sum_of_distances < .00001:
        # print "g1"
        # g1.print_graph()
        # print "g2"
        # g2.print_graph()
        # print "matching vertices", len(matching_vertices)
        # for (v1, v2) in matching_vertices:
        #     print "v1", v1.print_out(), "; v2", v2.print_out(), "; dist= ", v1.EuclidDist(v2)
        # print "end printout of matching vertices"
    max_possible_matches = min(g1.numVertices(), g2.numVertices())
    num_matching = float(max(len(matching_vertices), .001))
    
    # weigh the distance deviations 
    # using sigmoid function to return high weight unless over 20 matching--the lower the weight the better 
    scale_factor = ( max_vertices / num_matching ) 
    penalty = 1+10/(1+math.exp(num_matching - max_vertices/2)) 
    
    # print "SF=", scale_factor, ", sum=", sum_of_distances, ", penalty=", penalty, " final=", scale_factor * penalty * sum_of_distances
    
    return scale_factor * penalty * sum_of_distances

def all_to_all_classification(all_char_indices, all_person_indices, test_char_indices, test_person_indices, graph_pool, threshold=30):
    correct_classifications = defaultdict(int)

    # evaluate fitness between one graph and all other graphs
    for test_char_index in test_char_indices:
        for test_person_index in test_person_indices:
            test_graph = graph_pool[test_char_index][test_person_index]
            closest_char = None
            min_fitness_val = sys.maxint

            print 'classifying character', test_char_index, 'by person', test_person_index
            
            # compare to all graphs in the pool
            for char_index in all_char_indices:
                for person_index in all_person_indices:
                    # make sure it's not compared to the original graph or a crossover from the original graph
                    if not (test_char_index == char_index and str(test_person_index) in str(person_index)):
                        comparison_graph = graph_pool[char_index][person_index]

                        # find closest matching characters by minimizing fitness function
                        fitness = fitness_between_nodes(test_graph, comparison_graph, threshold)
                        if fitness < min_fitness_val:
                            min_fitness_val = fitness
                            closest_char = char_index
                        print person_index, char_index, fitness
            print 'classification of character', test_char_index, 'by person', \
                    test_person_index, 'is', closest_char, 'with fitness', min_fitness_val
            print "\n\n"
            if closest_char == test_char_index:
                correct_classifications[test_char_index] += 1

    print "Summary of correct classifications: "
    for char,num_correct in correct_classifications.items():
        print char, num_correct, num_correct / float(len(test_person_indices))
