
import time
import numpy as np
import itertools
from typing import List, Tuple, Dict, Callable, Iterable, Union

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