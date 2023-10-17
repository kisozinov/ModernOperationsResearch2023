import random
import os
import time

class ColoringProblem:

    def __init__(self) -> None:
        self._neighbour_sets = list()
        self._maxcolor = 1
        self._colors = list()

    def read_graph_file(self, filename: str):
        with open(filename, 'r') as f:
            for line in f.readlines():
                if line[0] == 'c':
                    continue
                elif line[0] == 'p':
                    vertices = int(line.split()[2])
                    self._neighbour_sets = [set() for i in range(vertices)]
                    self._colors = [0 for i in range(vertices)]
                elif line[0] == 'e':
                    line_ = line.split()
                    start, finish = int(line_[1]), int(line_[2])
                    self._neighbour_sets[start-1].add(finish-1)
                    self._neighbour_sets[finish-1].add(start-1)
    
    def mini_shuffle(self, degrees_sorted):
        splitted_vertices = []
        temp = []
        for i in range(len(degrees_sorted)-1):
            temp.append(degrees_sorted[i]) 
            if degrees_sorted[i][1] != degrees_sorted[i+1][1]:
                splitted_vertices.append(temp)
                temp = []
        
        if not splitted_vertices:
            return degrees_sorted
        
        if splitted_vertices[-1][-1][1] != degrees_sorted[-1][1]:
            splitted_vertices.append(temp + [degrees_sorted[-1]])
        else:
            splitted_vertices[-1].append(degrees_sorted[-1])
        
        for sublist in splitted_vertices:
            random.shuffle(sublist)
        
        flattened_vertices = [item for sublist in splitted_vertices for item in sublist]
        return flattened_vertices

    def improved_coloring(self, vertex):
        color_count = [0] * (self._maxcolor + 2)

        for neighbour in self._neighbour_sets[vertex]:
            neighbour_color = self._colors[neighbour]
            if neighbour_color <= self._maxcolor:
                color_count[neighbour_color] = 1

        for color in range(1, self._maxcolor + 2):  # Увеличиваем максимальный цвет на 1
            if color_count[color] == 0:
                self._colors[vertex] = color
                if color > self._maxcolor:
                    self._maxcolor = color
                break

    def coloring(self, vertex):
        color_count = [0] * (self._maxcolor + 1)
        min_color = self._maxcolor + 1

        for neighbour in self._neighbour_sets[vertex]:
            neigbour_color = self._colors[neighbour]
            color_count[neigbour_color] += 1
        for color in range(1, self._maxcolor + 1):
            if color_count[color] == 0:
                min_color = color
                break
        
        self._colors[vertex] = min_color
        if min_color > self._maxcolor:
            self._maxcolor = min_color

    def GetSolution(self):
        degrees = [len(n) for n in self._neighbour_sets]
        degrees_sorted = sorted(enumerate(degrees), key=lambda i: i[1], reverse=True)
        degrees_sorted = self.mini_shuffle(degrees_sorted)
        while degrees_sorted:
            self.improved_coloring(degrees_sorted[0][0])
            degrees_sorted.pop(0)
            # degrees_sorted = self.mini_shuffle(degrees_sorted)
        
    
    def GreedyGraphColoring(self, iters: int = 80):
        self.best_result = 10 ** 8
        for i in range(iters):
            self.GetSolution()
            if self.num_colors < self.best_result:
                self.best_result = self.num_colors
            if i != iters-1:
                self._maxcolor = 1
                self._colors = [0 for i in range(len(self._neighbour_sets))]

    def Check(self) -> bool:
        for i in range(len(self._neighbour_sets)):
            if self._colors[i] == 0:
                print(f"vertex {i+1} is not colored!")
                return False
            for neighbour in self._neighbour_sets[i]:
                if self._colors[neighbour] == self._colors[i]:
                    print(f"Neighbour vertices {i+1} and {neighbour + 1} have the same color!")
                    return False
        return True

    @property
    def num_colors(self):
        return self._maxcolor
    
    @property
    def colors(self):
        return self.colors
    

def main():
    files = ["myciel3.col", "myciel7.col", "school1.col", "school1_nsh.col", "anna.col", "miles1000.col", "miles1500.col", "le450_5a.col", "le450_15b.col", "queen11_11.col"]
    for file in files:
        problem = ColoringProblem()
        problem.read_graph_file(os.getcwd() + '\\data\\' + file)
        start = time.time()
        problem.GreedyGraphColoring()
        total_time = time.time() - start
        print(file)
        if not problem.Check():
            print("ERROR: Incorrect coloring")
        print(f"total time: {total_time:.4f}" )
        print(problem.best_result, end="\n======================\n")

if __name__ == "__main__":
    main()
