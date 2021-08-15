from context import Camera
import sympy as sp

t = sp.Symbol('t')
camera = Camera(filepath='./trajs/mandala0_gt.txt', max_vals=5)