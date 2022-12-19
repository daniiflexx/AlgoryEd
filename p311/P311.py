import time
import numpy as np
import matplotlib.pyplot as plt

from typing import List, Tuple, Dict, Callable, Union

def split(t: List) -> tuple[List, int, List]:
    """ Función que devuelve una tupla con las listas de elementos menores y mayores que el pivote, además del pivote
    Args:
        t (List): lista con la tabla original
    Returns: 
        tuple[List, int, List]: lista de menores, el pivote, lista de mayores
    """
    mid = t[0]
    t_l = [u for u in t if u < mid]
    t_r = [u for u in t if u > mid]
    return (t_l, mid, t_r)

def qsel(t: np.ndarray, k: int)-> Union[int, None]:
    """ Función que devuelve el elemento que corresponde al índice de la tabla ordenada
    Args:
        t (np.ndarray): la tabla ordenada
        k: el índice del elemento a devolver
    Returns: 
        Union(int, None): elemento a devolver
    """
    if k >= len(t) or k < 0:
        return
    if len(t) == 1 and k == 0:
        return t[0]

    t_l, mid, t_r = split(t)
    m = len(t_l)
    if k == m:
        return mid
    elif k < m:
        return qsel(t_l, k)
    else:
        return qsel(t_r, k-m-1)

def qsel_nr(t: np.ndarray, k: int)-> Union[int, None]:
    """ Función que devuelve el elemento que corresponde al índice de la tabla ordenada sin recursividad
    Args:
        t (np.ndarray): la tabla ordenada
        k: el índice del elemento a devolver
    Returns: 
        Union(int, None): elemento a devolver
    """

    tt = t.copy()
    while len(tt) > 0:
        if len(tt) == 1 and k == 0:
            return tt[0]
        t_l, mid, t_r = split(tt)
        m = len(t_l)
        if k == m:
            return mid
        elif k < m:
            tt = t_l
        else:
            tt = t_r
            k = k-m-1
    return

def split_pivot(t: np.ndarray, mid: int)-> Tuple[np.ndarray, int, np.ndarray]:
    """ Función que devuelve una tupla con las listas de elementos menores y mayores que el pivote, además del pivote
    Args:
        t (np.ndarray): lista con la tabla original
    Returns: 
        tuple[np.ndarray, int, np.ndarray]: lista de menores, el pivote, lista de mayores
    """

    s1 = [i for i in t if i < mid]
    s2 = [i for i in t if i > mid]
    return s1, mid, s2

def pivot5(t: np.ndarray)-> int:
    """ Función que escoge el mejor pivote(mediana de medianas)
    Args:
        t (np.ndarray): lista con la tabla original
    Returns: 
        int: pivote
    """

    if len(t) > 5:
        group_size = 5
        n_group = len(t) // group_size
        index_median = group_size // 2 
        
        sublists =  [t[j:j+ group_size] for j in range(0, len(t), group_size)][:n_group]
        
        medians = [sorted(sub)[index_median] for sub in sublists]
        
        if len(medians)%2 == 0:
            pivot = sorted(medians)[len(medians)//2-1]
        else:
            pivot = sorted(medians)[len(medians)//2]
            
        return pivot
    else:
        return np.sort(t)[len(t)//2]

def qsel5_nr(t: np.ndarray, k: int)-> Union[int, None]:
    """ Función que devuelve el elemento que corresponde al índice de la tabla ordenada sin recursividad, usando pivot5
    Args:
        t (np.ndarray): la tabla ordenada
        k: el índice del elemento a devolver
    Returns: 
        Union(int, None): elemento a devolver
    """

    if k >= len(t) or k < 0:
        return
    
    tt = t.copy()
    while len(tt) > 5:

        mid = pivot5(np.array(tt))

        t_l, mid, t_r = split_pivot(tt, mid)
        m = len(t_l)

        if k == m:
            return mid
        elif k < m:
            tt = t_l
        else:
            tt = t_r
            k = k-m-1
    if len(tt) <= 5:
        return int(np.sort(tt)[k])

def qsort_5(t: np.ndarray)-> np.ndarray:
    """ Función quicksort para ordenar una tabla, usando pivot5
    Args:
        t (np.ndarray): la tabla a ordenar
    Returns: 
        np.ndarray: la tabla ordenada
    """
    
    if t is None:
        return None

    if len(t) == 0:
        return np.array([])
    
    x = t.copy()
    mid = pivot5(np.array(x))

    t_left, mid, t_right = split_pivot(x, mid)

    if len(t_left) > 1:
        t_left = qsort_5(t_left)
    if len(t_right) > 1:
        t_right = qsort_5(t_right)

    t_left = np.append(t_left, np.array([mid]))

    return np.append(t_left, t_right)

def edit_distance(str_1: str, str_2: str)-> int:
	"""Función que devuelve el mínimo número de cambios para convertir str1 en str2
    Args:
        str_1: primera palabra
		str_2: segunda palabra
    Returns: 
        dist_matrix[n_rows - 1][n_columns - 1]: número mínimo de cambios
    """
	n_rows, n_columns = 1+len(str_1), 1+len(str_2)

	dist_matrix = np.zeros((n_rows, n_columns)).astype(int)
	dist_matrix[0] = np.arange(n_columns).astype(int)
	dist_matrix[ : , 0] = np.arange(n_rows).astype(int)

	for i in range(1, n_rows):
		for j in range(1, n_columns):
			if str_1[i-1] == str_2[j-1]:
				dist_matrix[i, j] = dist_matrix[i-1, j-1]
			else:
				dist_matrix[i, j] = 1 + min(dist_matrix[i-1, j-1], dist_matrix[i-1, j], dist_matrix[i, j-1])

	return dist_matrix[n_rows - 1][n_columns - 1]

def max_subsequence_length(str_1: str, str_2: str)-> int:
	"""Función que devuelve la longitud de la máxima subsequencia común entre str1 y str2
    Args:
        str_1: primera palabra
		str_2: segunda palabra
    Returns: 
        e[len(str_1), len(str_2)]: longitud de la subsequencia
    """
	e = np.zeros((len(str_1)+1, len(str_2)+1), dtype=int)
    
	for i in range(1, len(str_1)+1):
		for j in range(1, len(str_2)+1):
			if (str_1[i-1] == str_2[j-1]):
				e[i,j] = 1 + e[i-1, j-1]    
			else :
				e[i, j] = max(e[i-1, j], e[i, j-1])
	return e[len(str_1), len(str_2)]

def max_common_subsequence(str_1: str, str_2: str)-> str:
	"""Función que devuelve la máxima subsequencia común entre str1 y str2
    Args:
        str_1: primera palabra
		str_2: segunda palabra
    Returns: 
       	word: subsequencia
    """
	e = np.zeros((len(str_1)+1, len(str_2)+1), dtype=int)
	word = ""
	z = 0
    
	for i in range(1, len(str_1)+1):
		for j in range(1, len(str_2)+1):
			if (str_1[i-1] == str_2[j-1]):
				if e[i-1, j-1] == z:
					z += 1
					word += str_1[i-1]
				e[i,j] = 1 + e[i-1, j-1]    
			else :
				e[i, j] = max(e[i-1, j], e[i, j-1])
	return word

def min_mult_matrix(l_dims: List[int])-> int:
    """Función que devuelve el mínimo número de productos para multiplicar n matrices
    Args:
        l_dims (List[int]): lista con las dimensiones de las matrices
    Returns: 
        m: número de productos
    """
    m = np.inf * np.ones((len(l_dims)-1, len(l_dims)-1))
    np.fill_diagonal(m, 0)

    x = len(l_dims) - 1

    for j in range(0, x-1):
        m[j, j+1] = l_dims[j] * l_dims[j+1] * l_dims[j+2]

    for a in range(2, x):
        for b in range(x-a):
            for c in range(b, b+a):
                m[b, b+a] = min(m[b, b+a], l_dims[b] * l_dims[c+1] * l_dims[b+a+1] + m[b, c] + m[c+1, a+b])

    return m	