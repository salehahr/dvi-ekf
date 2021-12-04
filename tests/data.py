import sympy as sp
from context import Camera

t = sp.Symbol("t")
camera = Camera(filepath="./trajs/mandala0_gt.txt", max_vals=5)
