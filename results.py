#!/usr/bin/env python
from bmp_to_graphs import produce_graphs
from classify import fitness_between_nodes
from crossover import generate_crossovers_from_graphs
import matplotlib.pyplot as plt

# Use this file for pulling out results. 

def generate_fitness_plot(specific_char, specific_person, char_indices, person_indices, threshold, doCrossover = False):
    # produce graphs for each character
    graphs = produce_graphs(char_indices, person_indices, "coords")
    if doCrossover:
    	generate_crossovers_from_graphs(graphs, char_indices, person_indices)
    
    test_graph = graphs[specific_char][specific_person]
    print 'classifying character', specific_char, 'by person', specific_person
    fitnesses = []
    # classify character test_char by test_person
    for char_index in char_indices:
        for person_index in person_indices:
            if not (char_index == specific_char and person_index == specific_person and str(specific_person) in str(person_index)):
                g1 = test_graph
                g2 = graphs[char_index][person_index]

                # find closest matching characters by minimizing fitness function
                fitness = fitness_between_nodes(g1, g2, threshold)
                fitnesses.append((char_index, fitness))
                print person_index, char_index, fitness
    fzip = zip(*fitnesses)
    return fzip

def make_fitness_plot(doCrossover=False):
	c_chars = [9, 11, 12]
	c_person = 1
	char_list = [11, 12, 9]
	person_list = range(1,5)
	threshold = 30

	for char in char_list:
		fits = generate_fitness_plot(char, c_person, char_list, person_list, threshold, doCrossover)
		x = list(fits[0])
		y = list(fits[1])
		plt.title("Char " + str(char) + " Comparison")
		if doCrossover:
			plt.title("Char " + str(char) + " Comparison with Crossover")
		plt.xlabel("Character Index")
		plt.ylabel("Fitness")
		plt.scatter(x, y)
		# plt.semilogy(1-y)
		plt.yscale("log")
		plt.axis(xmin = min(char_list)-1, xmax = max(char_list) + 1, ymin=0, ymax=1000000)
		plt.savefig("writeup/char_" + str(char)+ "_classification")
		#plt.show()


if __name__ == "__main__":
    make_fitness_plot(True)


