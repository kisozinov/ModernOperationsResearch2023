import random
import os
import time
from collections import Counter

class MaxCliqueProblem:

    def __init__(self) -> None:
        self._neighbour_sets = list()
        self._best_clique = list()

    def read_graph_file(self, filename: str):
        with open(filename, 'r') as f:
            for line in f.readlines():
                if line[0] == 'c':
                    continue
                elif line[0] == 'p':
                    vertices = int(line.split()[2])
                    self._neighbour_sets = [set() for i in range(vertices)]
                elif line[0] == 'e':
                    line_ = line.split()
                    start, finish = int(line_[1]), int(line_[2])
                    self._neighbour_sets[start-1].add(finish-1)
                    self._neighbour_sets[finish-1].add(start-1)
    
    def Check(self) -> bool:
        if len(self._best_clique) != len(set(self._best_clique)):
            print("ERROR: Duplicated vertices in the clique")
            return False
        
        for i in self._best_clique:
            for j in self._best_clique:
                if i != j and Counter(self._neighbour_sets[i])[j] == 0:
                    print("ERROR: Returned subgraph is not a clique")
                    return False
        return True
    
    def FindClique(self, iterations):
        for i in range(iterations):
            current_clique = []

            vertices = list(range(len(self._neighbour_sets)))

            while vertices:
                random_vertex = random.choice(vertices)

                if all(random_vertex in self._neighbour_sets[vertex] for vertex in current_clique):
                    current_clique.append(random_vertex)

                vertices.remove(random_vertex)
            # print(len(current_clique))
            if len(current_clique) > len(self._best_clique):
                self._best_clique = current_clique

    @property
    def best_clique(self):
        return self._best_clique


def main():
    files = ["brock200_1", "brock200_2", "brock200_3", "brock200_4", "brock400_1", "brock400_2", "brock400_3", "brock400_4", "C125.9", "gen200_p0.9_44", "gen200_p0.9_55", "hamming8-4", "johnson16-2-4", "johnson8-2-4", "keller4", "MANN_a27", "MANN_a9", "p_hat1000-1", "p_hat1000-2", "p_hat1500-1", "p_hat300-3", "p_hat500-3", "san1000", "sanr200_0.9", "sanr400_0.7"]
    for file in files:
        problem = MaxCliqueProblem()
        problem.read_graph_file(os.getcwd() + '\\data\\' + file + ".clq")
        start = time.time()
        problem.FindClique(iterations=20000)
        total_time = time.time() - start
        print(file)
        if not problem.Check():
            print("ERROR: Incorrect clique")
        print(f"total time: {total_time:.4f}" )
        print(len(problem.best_clique), end="\n======================\n")

if __name__ == "__main__":
    main()    
