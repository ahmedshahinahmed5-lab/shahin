import streamlit as st
import sympy as sp
import re

# =========================================================
# SYMBOLS
# =========================================================
x, y, t = sp.symbols('x y t')
dy = sp.symbols('dy')
dx = sp.symbols('dx')

# =========================================================
# FUNCTIONS FROM YOUR TKINTER CODE (1:1)
# =========================================================

def standardize_eq(eq_str):
    eq_str = eq_str.replace(" ", "")
    if "dy/dx" in eq_str:
        return eq_str
    if "dy" in eq_str and "dx" in eq_str and "=" in eq_str:
        try:
            lhs, rhs = eq_str.split("=")
            lhs_expr = sp.sympify(lhs.replace("dy","1*dy"))
            rhs_expr = sp.sympify(rhs.replace("dx","1*dx"))
            dy_coeff = lhs_expr.coeff(dy)
            dx_coeff = rhs_expr.coeff(dx)
            if dy_coeff == 0 or dx_coeff == 0:
                return None
            f_expr = dx_coeff / dy_coeff
            return f"dy/dx = {sp.simplify(f_expr)}"
        except:
            return None
    return None

def analyze_input(eq_str):
    eq_str = eq_str.strip()
    if "=" not in eq_str:
        return False, False, "❌ Not an equation", None
    if "dx" in eq_str and "dy" in eq_str:
        return True, True, "✅ ODE (Differential Equation)", eq_str
    
    eq_std = standardize_eq(eq_str)
    if eq_std is None:
        return True, False, "⚠️ Equation but NOT ODE", None
    return True, True, "✅ ODE (Differential Equation)", eq_std

def parse_equation(eq_str):
    eq_str = eq_str.replace(" ", "").replace("dy/dx", "dy")
    lhs, rhs = eq_str.split("=")
    lhs = sp.sympify(lhs)
    rhs = sp.sympify(rhs)
    return sp.Eq(lhs, rhs)

def extract_f(eq):
    try:
        sol = sp.solve(eq, dy, dict=True)
        if not sol or len(sol) == 0:
            return None
        return sp.simplify(sol[0][dy])
    except:
        return None

def is_separable(f):
    try:
        f = sp.simplify(f)
        if f.is_Mul:
            fac = f.as_ordered_factors()
            has_x = any(fi.has(x) for fi in fac)
            has_y = any(fi.has(y) for fi in fac)
            if has_x and has_y:
                return True
        fx = f.subs(y,1)
        fy = sp.simplify(f/fx)
        if not fx.has(y) and not fy.has(x):
            return True
        return False
    except:
        return False

def is_bernoulli(f):
    try:
        terms = sp.expand(f).as_ordered_terms()
        powers=set()
        for term in terms:
            if term.has(y):
                p = sp.degree(term,y)
                if p is not None:
                    powers.add(p)
        if len(powers)>=2 and 1 in powers:
            return any(p not in (0,1) for p in powers)
        return False
    except:
        return False

def is_linear(f):
    try:
        a = sp.diff(f,y)
        if a.has(y):
            return False
        b = sp.simplify(f - a*y)
        if b.has(y):
            return False
        return True
    except:
        return False

def is_homogeneous_equation(eq):
    try:
        if isinstance(eq, str):
            if "=" in eq:
                lhs, rhs = eq.split("=")
                expr = sp.sympify(rhs)
            else:
                expr = sp.sympify(eq)
        else:
            expr = eq
        expr = sp.simplify(expr)
        new_expr = expr.subs({x: t*x, y: t*y})
        if sp.simplify(new_expr - expr) == 0:
            return "✅ Homogeneous"
        return ""
    except:
        return ""

def is_exact_equation(eq_str):
    try:
        eq_str = eq_str.replace(" ", "")
        if "dx" not in eq_str or "dy" not in eq_str or "=" not in eq_str:
            return ""
        lhs, rhs = eq_str.split("=")
        lhs_expr = sp.sympify(lhs.replace("dx","*dx").replace("dy","*dy"))
        rhs_expr = sp.sympify(rhs.replace("dx","*dx").replace("dy","*dy"))
        expr = lhs_expr - rhs_expr
        M = expr.coeff(dx)
        N = expr.coeff(dy)
        dM_dy = sp.diff(M, y)
        dN_dx = sp.diff(N, x)
        if sp.simplify(dM_dy - dN_dx) == 0:
            return "✅ Exact Equation"
        else:
            return "❌ Not Exact"
    except:
        return ""

def classify_equation(eq_str):
    valid, is_ode, msg, eq_std = analyze_input(eq_str)
    if not valid:
        return msg
    if not is_ode:
        return msg

    exact_check = is_exact_equation(eq_str)
    if exact_check == "✅ Exact Equation":
        return exact_check

    eq = parse_equation(eq_std)
    f = extract_f(eq)
    if f is None:
        return "❌ Cannot solve for dy/dx"

    results = []

    if is_bernoulli(f):
        results.append("✅ Bernoulli\nf(x,y) = " + str(f))

    if is_separable(f):
        results.append("✅ Separable\nf(x,y) = " + str(f))

    if is_linear(f):
        results.append("✅ Linear\nf(x,y) = " + str(f))

    h = is_homogeneous_equation(eq_std)
    if h:
        results.append(h)

    if results:
        return "\n".join(results)
    return "⚠️ Equation but classification not found"

# =========================================================
# STREAMLIT UI
# =========================================================
st.set_page_config(page_title="ODE Analyzer", page_icon="📘")
st.title("📘 Ordinary Differential Equation Solver — Streamlit Version")

eq_input = st.text_input("Enter equation:", placeholder="Example: dy/dx + y = x*y**2")

if st.button("Solve ✅"):
    result = classify_equation(eq_input)
    st.success("✅ Result:")
    st.code(result)





# =========================================================================================ظظظ
