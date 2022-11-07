import numpy as np
import itertools as it
from typing import List, Dict, Callable, Iterable
import sys as sys

def init_cd(n: int)-> np.ndarray:
    lst = n * [-1]
    arr = np.array(lst)
    return arr
def find(ind: int, p_cd: np.ndarray)-> int:
    z = ind

    while p_cd[z] > -1:
        z = p_cd[z]
        
    while p_cd[ind] >-1:
        y = p_cd[ind]
        p_cd[ind] = z
        ind = y
    return z

def union(rep_1: int, rep_2: int, p_cd: np.ndarray)-> int:
    x = find(rep_1, p_cd)
    y = find(rep_2, p_cd)
        
    if x == y:
            return -1
        
    if p_cd[y] < p_cd[x]:      
        p_cd[x] = y
        rep = y  
    elif p_cd[y] > p_cd[x]:    
        p_cd[y] = x 
        rep = x
    else:
        p_cd[y] = x
        p_cd[x] -= 1
        rep =  x     
    return rep

def cd_2_dict(p_cd: np.ndarray)-> dict:
    d = {}
    for i in range(len(p_cd)):
        if p_cd[i] < 0:
            d[i] = []
        
    for i in range(len(p_cd)):
        aux = find(i, p_cd)
        d[aux].append(i)
    return d

def ccs(n: int, l: list)-> dict:
    cd = init_cd(n)
    for i in l:
        union(i[0], i[1], cd)
    d = cd_2_dict(cd)
    return d

def dist_matrix(n_nodes: int, w_max=10)-> np.ndarray:
    m = np.zeros((n_nodes, n_nodes), dtype=int)
    i = 0
    while i < n_nodes:
        j = i
        while j < n_nodes:
            if i != j:
                m[i][j] = np.random.randint(0, w_max+1)
                m[j][i] = m[i][j]
            j+=1
        i+=1
    return(m)

def greedy_tsp(dist_m: np.ndarray, node_ini=0)-> list:
    num_nodes = dist_m.shape[0]
    circuit = [node_ini]
    while len(circuit) < num_nodes:
        current_city = circuit[-1]
        options = np.argsort(dist_m[ current_city])
        for i in options:
            if i not in circuit:
                circuit.append(i)
                break
    circuit.append(node_ini)
    return circuit

def len_circuit(circuit: list, dist_m: np.ndarray)-> int:
    i = 0
    longitud = 0
    while (i < len(circuit) - 1):
        longitud += dist_m[circuit[i]][circuit[i+1]]
        i+=1
    return longitud

def repeated_greedy_tsp(dist_m: np.ndarray)-> list:
    num_nodes = dist_m.shape[0]
    i = 0
    longitud = sys.maxsize
    while i < num_nodes:
        circuit = greedy_tsp(dist_m, i)
        aux = len_circuit(circuit, dist_m)
        if aux < longitud:
            longitud = aux
            circuito = list(circuit)
        i+=1
    return circuito

def exhaustive_tsp(dist_m: np.ndarray)-> list:
    circuito = []
    size = dist_m.shape[0]
    p = it.permutations(list(range(size)))
    longitud = sys.maxsize
    for circuit in p:
        circuit = list(circuit)
        aux = len_circuit(circuit, dist_m)
        circuit.append(circuit[0])
        if aux < longitud:
            longitud = aux
            circuito = list(circuit)
    return circuito
