import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from symbolics import symbols as syms
from symbolics import equations as eqns
from symbolics import functions as symfcns

from Filter import VisualTraj, Quaternion