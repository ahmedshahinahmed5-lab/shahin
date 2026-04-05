import streamlit as st
import sympy as sp

x, y, t, C1 = sp.symbols('x y t C1')
dy = sp.symbols('dy')
dx = sp.symbols('dx')

# =========================================================
# HELPERS
# =========================================================

def standardize_eq(eq_str):
    eq_str = eq_str.replace(" ", "")
    if "dy/dx" in eq_str:
        return eq_str
    if "dy" in eq_str and "dx" in eq_str and "=" in eq_str:
        try:
            lhs, rhs = eq_str.split("=")
            lhs_expr = sp.sympify(lhs.replace("dy", "1*dy"))
            rhs_expr = sp.sympify(rhs.replace("dx", "1*dx"))
            dy_coeff = lhs_expr.coeff(dy)
            dx_coeff = rhs_expr.coeff(dx)
            if dy_coeff == 0 or dx_coeff == 0:
                return None
            return f"dy/dx = {sp.simplify(dx_coeff / dy_coeff)}"
        except:
            return None
    return None

def analyze_input(eq_str):
    eq_str = eq_str.strip()
    if "=" not in eq_str:
        return False, False, "❌ Not an equation", None
    if "dx" in eq_str and "dy" in eq_str:
        return True, True, "✅ ODE", eq_str
    eq_std = standardize_eq(eq_str)
    if eq_std is None:
        return True, False, "⚠️ Not an ODE", None
    return True, True, "✅ ODE", eq_std

def parse_equation(eq_str):
    eq_str = eq_str.replace(" ", "").replace("dy/dx", "dy")
    lhs, rhs = eq_str.split("=")
    return sp.Eq(sp.sympify(lhs), sp.sympify(rhs))

def extract_f(eq):
    try:
        sol = sp.solve(eq, dy, dict=True)
        return sp.simplify(sol[0][dy]) if sol else None
    except:
        return None

def get_MN(eq_str):
    eq_str = eq_str.replace(" ", "")
    lhs, rhs = eq_str.split("=")
    lhs_e = sp.sympify(lhs.replace("dx", "*dx").replace("dy", "*dy"))
    rhs_e = sp.sympify(rhs.replace("dx", "*dx").replace("dy", "*dy"))
    expr = lhs_e - rhs_e
    return expr.coeff(dx), expr.coeff(dy)

# =========================================================
# CLASSIFIERS
# =========================================================

def is_exact(eq_str):
    try:
        if "dx" not in eq_str or "dy" not in eq_str or "=" not in eq_str:
            return False
        M, N = get_MN(eq_str)
        return sp.simplify(sp.diff(M, y) - sp.diff(N, x)) == 0
    except:
        return False

def is_linear(f):
    try:
        a = sp.diff(f, y)
        if a.has(y): return False
        return not sp.simplify(f - a * y).has(y)
    except:
        return False

def is_separable(f):
    try:
        f = sp.simplify(f)
        if f.is_Mul:
            facs = f.as_ordered_factors()
            if any(fi.has(x) for fi in facs) and any(fi.has(y) for fi in facs):
                return True
        fx = f.subs(y, 1)
        fy = sp.simplify(f / fx)
        return not fx.has(y) and not fy.has(x)
    except:
        return False

def is_bernoulli(f):
    try:
        powers = set()
        for term in sp.expand(f).as_ordered_terms():
            if term.has(y):
                p = sp.degree(term, y)
                if p is not None:
                    powers.add(p)
        return len(powers) >= 2 and 1 in powers and any(p not in (0, 1) for p in powers)
    except:
        return False

def is_homogeneous(f):
    try:
        ratio = sp.simplify(f.subs({x: t*x, y: t*y}) / f)
        return not ratio.has(t)
    except:
        return False

# =========================================================
# SOLVERS
# =========================================================

def solve_linear(f):
    try:
        P = -sp.diff(f, y)
        Q = sp.simplify(f + P * y)
        mu = sp.exp(sp.integrate(P, x))
        return f"y = {sp.simplify((sp.integrate(Q * mu, x) + C1) / mu)}"
    except Exception as e:
        return f"❌ {e}"

def solve_separable(f):
    try:
        g = f.subs(y, 1)
        h = sp.simplify(1 / (f / g))
        return f"{sp.integrate(h, y)} = {sp.integrate(g, x)} + C"
    except Exception as e:
        return f"❌ {e}"

def solve_bernoulli(f):
    try:
        powers = {}
        for term in sp.expand(f).as_ordered_terms():
            if term.has(y):
                p = sp.degree(term, y)
                powers[p] = powers.get(p, sp.Integer(0)) + sp.simplify(term / y**p)
        n = max(p for p in powers if p not in (0, 1))
        P_c = -powers.get(1, sp.Integer(0))
        Q_c = powers.get(n, sp.Integer(0))
        mu = sp.exp(sp.integrate((1 - n) * P_c, x))
        v_sol = sp.simplify((sp.integrate((1 - n) * Q_c * mu, x) + C1) / mu)
        return f"y = {sp.simplify(v_sol ** sp.Rational(1, 1-n))}   [v = y^{1-n}, n={n}]"
    except Exception as e:
        return f"❌ {e}"

def solve_homogeneous(f):
    try:
        v = sp.Function('v')(x)
        rhs = sp.simplify(f.subs(y, v * x) - v)
        if rhs == 0:
            return "y = C·x"
        return f"{sp.integrate(1/rhs, v)} = {sp.integrate(1/x, x)} + C   [v = y/x]"
    except Exception as e:
        return f"❌ {e}"

def solve_exact(eq_str):
    try:
        M, N = get_MN(eq_str)
        F = sp.integrate(M, x)
        g = sp.integrate(sp.simplify(N - sp.diff(F, y)), y)
        return f"F(x,y) = {sp.simplify(F + g)} = C"
    except Exception as e:
        return f"❌ {e}"

# =========================================================
# CLASSIFY
# =========================================================

def classify_equation(eq_str):
    valid, is_ode, msg, eq_std = analyze_input(eq_str)
    if not valid or not is_ode:
        return msg, {}

    types_found = {}

    if is_exact(eq_str):
        types_found["Exact"] = True

    eq = parse_equation(eq_std)
    f = extract_f(eq)
    if f is None:
        return "❌ Cannot extract dy/dx", {}

    if is_bernoulli(f): types_found["Bernoulli"] = f
    if is_separable(f):  types_found["Separable"] = f
    if is_linear(f):     types_found["Linear"] = f
    if is_homogeneous(f): types_found["Homogeneous"] = f

    if not types_found:
        return "⚠️ Type not recognised", {}

    return "✅ Detected: " + ", ".join(types_found.keys()), types_found

# =========================================================
# UI
# =========================================================

st.set_page_config(page_title="ODE Solver", page_icon="🧮")
st.title("🧮 ODE Solver")

# session state init
for key in ("types_found", "label", "solution", "sol_type", "eq_saved"):
    if key not in st.session_state:
        st.session_state[key] = {} if key == "types_found" else ""

# ── Examples ──
with st.sidebar:
    st.header("📚 Examples")
    examples = {
        "Linear":       "dy/dx = -2*y + x",
        "Separable":    "dy/dx = x*y",
        "Bernoulli":    "dy/dx = y + x*y**2",
        "Homogeneous":  "dy/dx = (x+y)/x",
        "Exact":        "(2*x*y)*dx + (x**2)*dy = 0",
    }
    for kind, eq in examples.items():
        if st.button(f"{kind}: {eq}", use_container_width=True):
            st.session_state.eq_saved = eq
            st.session_state.types_found = {}
            st.session_state.label = ""
            st.session_state.solution = ""
            st.rerun()

# ── Input ──
eq_input = st.text_input("Enter ODE:", value=st.session_state.eq_saved,
                         placeholder="dy/dx = x*y  or  M dx + N dy = 0")

col1, col2 = st.columns([1, 1])

with col1:
    if st.button("🔍 Analyze", use_container_width=True):
        if eq_input.strip():
            label, types = classify_equation(eq_input.strip())
            st.session_state.label = label
            st.session_state.types_found = types
            st.session_state.solution = ""
            st.session_state.eq_saved = eq_input.strip()

with col2:
    if st.button("🗑 Clear", use_container_width=True):
        for key in ("types_found", "label", "solution", "sol_type", "eq_saved"):
            st.session_state[key] = {} if key == "types_found" else ""
        st.rerun()

# ── Analysis result ──
if st.session_state.label:
    st.info(st.session_state.label)

# ── Solve buttons ──
types = st.session_state.types_found
if types:
    st.write("**Solve as:**")
    cols = st.columns(len(types))
    for i, (name, f_or_flag) in enumerate(types.items()):
        with cols[i]:
            if st.button(f"Solve {name}", use_container_width=True):
                eq_str = st.session_state.eq_saved
                if name == "Exact":        res = solve_exact(eq_str)
                elif name == "Linear":     res = solve_linear(f_or_flag)
                elif name == "Separable":  res = solve_separable(f_or_flag)
                elif name == "Bernoulli":  res = solve_bernoulli(f_or_flag)
                elif name == "Homogeneous": res = solve_homogeneous(f_or_flag)
                else: res = "❌ Unknown"
                st.session_state.solution = res
                st.session_state.sol_type = name

# ── Solution output ──
if st.session_state.solution:
    sol = st.session_state.solution
    stype = st.session_state.sol_type
    if sol.startswith("❌"):
        st.error(f"[{stype}] {sol}")
    else:
        st.success(f"[{stype} Solution]\n\n{sol}")
