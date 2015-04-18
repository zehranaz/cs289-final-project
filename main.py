from classes import Graph, Edge, Vertex
import matplotlib.pyplot as plt

def main():
    # graph = Graph()
 
    char_index = 11
    for person_index in range(1, 2):
        filename = "lao_images/000" + str(person_index) + "_" + str(char_index) + ".bmp"
        
    # read image as 2D numpy array
    im = plt.imread(filename)
    nrow, ncol = im.shape[0], im.shape[1]
    plt.show()

    # walk through image diagonally until see something black
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
    print 'current box', im[rowi][coli] # should be nonzero now
    print 'current index', rowi, coli


    r = 1 # radius around current pixel to be checked
    numiter = 10
    prev_visited = set()
    prev_visited.add((currentrow, currentcol))
    for i in range(numiter):
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
        currentrow = row_min
        currentcol = col_min
        prev_visited.add((currentrow, currentcol))

    


if __name__ == "__main__":
    main()
