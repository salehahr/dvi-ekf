import sympy as sp
from casadi import *


def sympy2casadi(sympy_expr, sympy_var, casadi_var):
    # run checks on sympy_var
    assert isinstance(sympy_var, list)
    is_sym_vars = [isinstance(s, sp.Expr) for s in sympy_var]
    if not any(is_sym_vars):
        # no symbolic variables in arg: sympy_var -- nothing to replace
        return casadi.SX(sympy_expr)

    assert casadi_var.is_vector()
    # convert to list
    if casadi_var.shape[1] > 1:
        casadi_var = casadi_var.T
    casadi_var = casadi.vertsplit(casadi_var)

    mapping = {
        "ImmutableDenseMatrix": casadi.blockcat,
        "MutableDenseMatrix": casadi.blockcat,
        "Abs": casadi.fabs,
    }
    f = sp.lambdify(sympy_var, sympy_expr, modules=[mapping, casadi])
    return f(*casadi_var)


def dummify_undefined_functions(expr):
    mapping = {}

    # replace all Derivative terms
    for der in expr.atoms(sp.Derivative):
        f_name = der.expr.func.__name__
        der_count = der.derivative_count
        ds = "d" * der_count
        # var_names = [var.name for var in der.variables]
        # name = "d%s_d%s" % (f_name, 'd'.join(var_names))
        name = f"{f_name}_{ds}ot"  # % (f_name, 'd'.join(var_names))
        mapping[der] = sp.Symbol(name)

    # replace undefined functions
    from sympy.core.function import AppliedUndef

    for f in expr.atoms(AppliedUndef):
        f_name = f.func.__name__
        mapping[f] = sp.Symbol(f_name)

    return expr.subs(mapping)


def dummify_array(expr):
    is_array = isinstance(expr, (list, np.ndarray))

    if is_array:
        for i, a in enumerate(expr):
            expr[i] = dummify_undefined_functions(a)
        return expr
    else:
        return dummify_undefined_functions(expr)
