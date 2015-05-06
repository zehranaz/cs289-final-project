from main import produce_graphs, fitness_between_nodes
import matplotlib.pyplot as plt

# Use this file for pulling out results. 

def generate_fitness_plot(specific_char, specific_person, char_indices, person_indices, threshold):
    
    # produce graphs for each character
    graphs = produce_graphs(char_indices, person_indices, "coords")
    test_graph = graphs[specific_char][specific_person-1]
    print 'classifying character', specific_char, 'by person', specific_person
    fitnesses = []
    # classify character test_char by test_person
    for char_index in char_indices:
        for person_index in person_indices:
            #if not (char_index == specific_char and person_index == specific_person):
                g1 = test_graph
                g2 = graphs[char_index][person_index-1]

                # find closest matching characters by minimizing fitness function
                fitness = fitness_between_nodes(g1, g2, threshold)
                fitnesses.append((char_index, fitness))
                print person_index, char_index, fitness
    fzip = zip(*fitnesses)
    # chars = list(fzip[0])
    # fits = list(fzip[1])
    # plt.scatter(chars, fits)
    # plt.semilogy()
    # plt.axis(xmin = min(char_indices)-1, xmax = max(char_indices) + 1, ymin = 0, ymax = 100000)
    # plt.spines['top'].set_visible(False)
    # plt.spines['right'].set_visible(False)
    # plt.title("Fitness Classified")
    # plt.xlabel("Character Index")
    # plt.ylabel("Fitness")
    # plt.show()
    return fzip

def make_fitness_plot():
	c_chars = [9, 11, 12, 15]
	c_person = 3
	char_list = [11,12,9, 15]
	person_list = range(1,5)
	threshold = 30
	index = 1
	for char in char_list:
		fits = generate_fitness_plot(char, c_person, char_list, person_list, threshold)
		x = list(fits[0])
		y = list(fits[1])
		# ax = fig.add_subplot(2, 2, index)
		# index += 1
		plt.title("Char " + str(char) + " Comparison")
		plt.xlabel("Character Index")
		plt.ylabel("Fitness")
		plt.scatter(x, y)
		# plt.semilogy(1-y)
		plt.yscale("log")
		plt.axis(xmin = min(char_list)-1, xmax = max(char_list) + 1)
		plt.show()

if __name__ == "__main__":
    make_fitness_plot()


