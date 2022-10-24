import numpy as np
from typing import List, Callable


def matrix_multiplication(m_1: np.ndarray, m_2: np.ndarray) -> np.ndarray:
    """Función para multiplicar matrices

    Args:
        m_1(np.ndarray): Matriz 1, primer factor\n
        m_2(np.ndarray): Matriz 2, segundo factor

    Returns:
        m_3(np.ndarray): Matriz producto de m_1 y m_2"""
    num_rows1, num_cols1 = m_1.shape
    num_rows2, num_cols2 = m_2.shape
    m_3 = np.ndarray(shape=(num_rows1, num_cols2), dtype=int)
    if num_cols1 == num_rows2:
        for i in range(0, num_rows1):
            for z in range(0, num_cols2):
                aux = 0
                for j in range(0, num_rows2):
                    aux += m_1[i][j] * m_2[j][z]
                m_3[i][z] = aux
        return (m_3)


def rec_bb(t: list, f: int, l: int, key: int) -> int:
    """Función recursiva de búsqueda binaria

    Args:
        t(list): Lista donde se busca\n
        f(int): Índice desde donde quieres buscar\n
        l(int): Índice hasta el que buscar\n
        key(int): Elemento a buscar

    Returns:
        (int): Index of key in t"""
    if f > l:
        return
    if f == l:
        if key == t[f]:
            return f
        else:
            return
    mid = (f + l) // 2
    if key == t[mid]:
        return mid
    elif key < t[mid]:
        return rec_bb(t, f, mid - 1, key)
    else:
        return rec_bb(t, mid + 1, l, key)


def bb(t: list, f: int, l: int, key: int) -> int:
    """Función iterativa de búsqueda binaria

    Args:
        t(list): Lista donde se busca\n
        f(int): Índice desde donde quieres buscar\n
        l(int): Índice hasta el que buscar\n
        key(int): Elemento a buscar

    Returns:
        (int): Index of key in t"""
    while f <= l:
        mid = (f + l) // 2
        if key == t[mid]:
            return mid
        elif key < t[mid]:
            l = mid - 1
        else:
            f = mid + 1


def _l(i):
    return 2 * i + 1


def _r(i):
    return 2 * i + 2


def _p(i):
    return (i - 1) // 2


def min_heapify(h: np.ndarray, i: int):
    """Función que aplica el método heapify

    Args:
        h(np.ndarray): Lista donde se aplica el método\n
        i(int): Índice de la lista donde se aplica el método"""
    swap = True

    while swap:
        minimum = i
        l = _l(i)
        r = _r(i)

        if l < len(h) and h[l] < h[i]:
            minimum = l
        if r < len(h) and h[r] < h[minimum]:
            minimum = r

        if minimum != i:
            h[i], h[minimum] = h[minimum], h[i]
            i = minimum
            swap = True
        else:
            swap = False


def insert_min_heap(h: np.ndarray, k: int) -> np.ndarray:
    """Inserta el elemento k en el min heap contenido eh h
    y devuelve el nuevo min heap

    Args:
        h(np.ndarray): Lista que contiene el min heap\n
        k(int): Elemento a insertar

    Returns:
        h(np.ndarray): Min heap con el elemento insertado"""
    h = np.append(h, k)
    i = len(h) - 1

    while i > 0 and h[_p(i)] > h[i]:
        h[_p(i)], h[i] = h[i], h[_p(i)]
        i = _p(i)

    return h


def create_min_heap(h: np.ndarray):
    """Crea un min heap sobre el array de Numpy h de manera in-place

    Args:
        h(np.ndarray): Lista donde crear el min heap"""
    i = _p(len(h) - 1)
    while i > -1:
        min_heapify(h, i)
        i -= 1


def pq_ini():
    """Inicializa una cola de prioridad vacía

    Returns:
        pq(np.ndarray): La cola inicializada"""
    pq = np.ndarray(shape=0, dtype=int)
    return pq


def pq_insert(h: np.ndarray, k: int) -> np.ndarray:
    """Inserta el elemnto k en la cola de prioridad h y la devuelve

    Args:
        h(np.ndarray): Lista que contiene la cola\n
        k(int): Elemento a insertar

    Returns:
        h(np.ndarray): Cola con el elemento insertado"""
    h = insert_min_heap(h, k)
    return h


def pq_remove(h: np.ndarray) -> tuple[int, np.ndarray]:
    """Elimina el elemento con menor prioridad en h
    y lo devuelve junto a la lista

    Args:
        h(np.ndarray): Lista que contiene la cola

    Returns:
        root(int): Elemento eliminado\n
        h(np.ndarray): Lista modificada"""
    if len(h) == 0:
        h = np.ndarray(shape=(0))
        return 0, h
    root = h[0]
    aux = h[len(h) - 1]
    h = np.delete(h, len(h) - 1, 0)
    if len(h):
        h[0] = aux

    min_heapify(h, 0)
    return root, h


def select_min_heap(h: np.ndarray, k: int) -> int:
    """Devuelve el elemento en la posición k
    si el min heap estuviese ordenado

    Args:
        h(np.ndarray): Lista desordenada (min heap)\n
        k(int): Posición a buscar si estuviese ordenado

    Returns:
        (int): Elemento en la posición k"""
    h2 = [i * 1 for i in h]
    i = 0
    create_min_heap(h2)
    while i < k - 1:
        e, h2 = pq_remove(h2)
        i += 1

    return(h2[0])
