import pickle
import os
from classes import CrossOver

# takes in a graph_name and writes out graph to graph_name.pkl
def save_graph_to_file(graph, graph_name):
    directory = "pkl_files"
    if not os.path.exists(directory):
        os.makedirs(directory)
    output = open(directory + "/" + graph_name +".pkl", 'wb')
    pickle.dump(graph, output)
    output.close()

# reads out graph from pickle file and returns it
def read_graph_from_file(filename):
    directory = "pkl_files"
    if os.path.isfile(directory + "/" + filename):
        pkl_file = open(directory + "/" + filename, 'rb')
        graph = pickle.load(pkl_file)
        pkl_file.close()
        return graph
    return None

def get_crossed_filename(p1index, p2index, char):
    return "000" + str(p1index) + "_000" + str(p2index) + "_" + str(char)

# Given a char and person index, return name of file
# e.g. get_name_for_file(12, 23) = 00023_12
def get_name_for_file(char_index, person_index):
    return "000" + str(person_index) + "_" + str(char_index)

# given a dictionary of graphs indexed by character index and then person index, append crossedOver graphs 
def generate_crossovers_from_graphs(graph_pool, char_index_pool, person_index_pool, threshold=20):
    num_persons = len(person_index_pool)
    for char in char_index_pool:
        for i, person_i in enumerate(person_index_pool):
            for j, person_j in enumerate(person_index_pool):
                # only do crossover once between each possible pair
                if j <= i:
                    continue
                # check if crossover has already been done and saved to file
                new_graph = read_graph_from_file(get_crossed_filename(i, j, char) + ".pkl")
                if new_graph == None:
                    new_graph = read_graph_from_file(get_crossed_filename(j, i, char) + ".pkl")
                    if new_graph == None:
                        # actually do crossover if can't be loaded from file
                        graph1 = graph_pool[char][person_i]
                        graph2 = graph_pool[char][person_j]
                        new_graph = CrossOver(graph1, graph2, threshold)
                        save_graph_to_file(new_graph, graph_name = get_crossed_filename(person_i, person_j, char))
                graph_pool[char][str(person_i) + "_" + str(person_j)] = new_graph
    return graph_pool