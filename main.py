#!/usr/bin/env python
# custom code
from classes import Graph, CrossOver
from bmp_to_graphs import produce_graphs
from classify import all_to_all_classification
from crossover import generate_crossovers_from_graphs

def main():
    # training set against which we classify
    char_indices = [11, 12, 9] # 5, 18
    person_indices = range(1, 5) # not zero-indexed
    
    # produce nested dictionary, access via graphs[char_ind][person_ind]
    graphs = produce_graphs(char_indices, person_indices, "coords")

    # add crossovers to graphs matrix
    graph_pool = generate_crossovers_from_graphs(graphs, char_indices, person_indices)

    ''' this check worked fine with 3 chars x 4 people graphs and crossovers
    # sanity check: check all graphs exist that we expect to and # of vertices
    for char in graph_pool:
        for pind in graph_pool[char]:
            print 'char', char, 'pind', pind, 'vertices', graphs[char][pind].numVertices()
    '''
    # characters to be classified - can be a subset of all_char_indices
    test_char_indices = char_indices 
    # characters to be compared against
    all_char_indices = char_indices

    # which people's handwriting of test_char_indices will be classified
    test_person_indices = person_indices
    # add the crossover indices to the pool of all_person_indices to be tested
    all_person_indices = []
    for char in graph_pool:
        for key in graph_pool[char]:
            if key not in all_person_indices:
                all_person_indices.append(key)

    all_to_all_classification(all_char_indices, all_person_indices, test_char_indices, test_person_indices, graph_pool)
    return

if __name__ == "__main__":
    main()

