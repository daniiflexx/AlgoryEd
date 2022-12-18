
import time
import numpy as np
import itertools
from typing import List, Tuple, Dict, Callable, Iterable, Union

def split(t: List) -> tuple[List, int, List]:
    mid = t[0]
    t_l = [u for u in t if u < mid]
    t_r = [u for u in t if u > mid]
    return (t_l, mid, t_r)

def qsel(t: np.ndarray, k: int)-> Union[int, None]:
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
    s1 = [i for i in t if i < mid]
    s2 = [i for i in t if i > mid]
    return s1, mid, s2

def pivot5(t: np.ndarray)-> int:
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