# coding: utf-8

import numpy as np
import random

def newMod(a: np.ndarray, b: int):
    """
    a%b を行う (但し、a が 1 以上なのに a%b が 0 になってしまうものに対しては適当な値を振る)

    [引数]
    - a: np.ndarray (整数)
    - b: int (特に 2 以上の自然数)

    [戻り値]
    ret: a の剰余
    """

    ret = a%b
    ret[(a != 0) & (ret == 0)] = random.randint(1, b-1)
    return ret