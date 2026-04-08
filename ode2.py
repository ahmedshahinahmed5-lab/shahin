import streamlit as st
import sympy as sp

# =========================================================
# SYMBOLS
# =========================================================
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
    return sp.Eq(sp.sympify(lhs), sp.sympify(rhs))

def extract_f(eq):
    try:
        sol = sp.solve(eq, dy, dict=True)
        if not sol:
            return None
        return sp.simplify(sol[0][dy])
    except:
        return None

def get_MN(eq_str):
    eq_str = eq_str.replace(" ", "")
    lhs, rhs = eq_str.split("=")
    lhs_expr = sp.sympify(lhs.replace("dx", "*dx").replace("dy", "*dy"))
    rhs_expr = sp.sympify(rhs.replace("dx", "*dx").replace("dy", "*dy"))
    expr = lhs_expr - rhs_expr
    return expr.coeff(dx), expr.coeff(dy)

# =========================================================
# CLASSIFIERS
# =========================================================

def is_separable(f):
    try:
        f = sp.simplify(f)
        if f.is_Mul:
            fac = f.as_ordered_factors()
            if any(fi.has(x) for fi in fac) and any(fi.has(y) for fi in fac):
                return True
        fx = f.subs(y, 1)
        fy = sp.simplify(f / fx)
        return not fx.has(y) and not fy.has(x)
    except:
        return False

def is_bernoulli(f):
    try:
        terms = sp.expand(f).as_ordered_terms()
        powers = set()
        for term in terms:
            if term.has(y):
                p = sp.degree(term, y)
                if p is not None:
                    powers.add(p)
        if len(powers) >= 2 and 1 in powers:
            return any(p not in (0, 1) for p in powers)
        return False
    except:
        return False

def is_linear(f):
    try:
        a = sp.diff(f, y)
        if a.has(y):
            return False
        b = sp.simplify(f - a * y)
        return not b.has(y)
    except:
        return False

def is_homogeneous(f):
    try:
        new_expr = f.subs({x: t * x, y: t * y})
        ratio = sp.simplify(new_expr / f)
        return not ratio.has(t)
    except:
        return False

def is_exact(eq_str):
    try:
        eq_str = eq_str.replace(" ", "")
        if "dx" not in eq_str or "dy" not in eq_str or "=" not in eq_str:
            return False
        M, N = get_MN(eq_str)
        return sp.simplify(sp.diff(M, y) - sp.diff(N, x)) == 0
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
        sol = sp.simplify((sp.integrate(Q * mu, x) + C1) / mu)
        return f"y = {sol}"
    except Exception as e:
        return f"❌ Could not solve: {e}"

def solve_separable(f):
    try:
        g = f.subs(y, 1)
        h = sp.simplify(1 / (f / g))
        lhs = sp.integrate(h, y)
        rhs = sp.integrate(g, x)
        return f"{lhs} = {rhs} + C"
    except Exception as e:
        return f"❌ Could not solve: {e}"

def solve_bernoulli(f):
    try:
        terms = sp.expand(f).as_ordered_terms()
        powers = {}
        for term in terms:
            if term.has(y):
                p = sp.degree(term, y)
                powers[p] = powers.get(p, sp.Integer(0)) + sp.simplify(term / y**p)
        n = max(p for p in powers if p not in (0, 1))
        P_coeff = -powers.get(1, sp.Integer(0))
        Q_coeff = powers.get(n, sp.Integer(0))
        new_P = (1 - n) * P_coeff
        new_Q = (1 - n) * Q_coeff
        mu = sp.exp(sp.integrate(new_P, x))
        v_sol = sp.simplify((sp.integrate(new_Q * mu, x) + C1) / mu)
        y_sol = sp.simplify(v_sol ** (sp.Rational(1, 1 - n)))
        return f"y = {y_sol}   (substitution: v = y^{1-n}, n = {n})"
    except Exception as e:
        return f"❌ Could not solve: {e}"

def solve_homogeneous(f):
    try:
        v = sp.Function('v')(x)
        f_sub = f.subs(y, v * x)
        F_v = sp.simplify(f_sub)
        rhs = sp.simplify(F_v - v)
        if rhs == 0:
            return "y = C·x"
        lhs_int = sp.integrate(1 / rhs, v)
        rhs_int = sp.integrate(1 / x, x)
        return f"{lhs_int} = {rhs_int} + C   (substitution: v = y/x)"
    except Exception as e:
        return f"❌ Could not solve: {e}"

def solve_exact(eq_str):
    try:
        M, N = get_MN(eq_str)
        F = sp.integrate(M, x)
        g_prime = sp.simplify(N - sp.diff(F, y))
        g = sp.integrate(g_prime, y)
        F_total = sp.simplify(F + g)
        return f"F(x, y) = {F_total} = C"
    except Exception as e:
        return f"❌ Could not solve: {e}"

# =========================================================
# CLASSIFY → returns dict of detected types + f
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

    if is_bernoulli(f):
        types_found["Bernoulli"] = f
    if is_separable(f):
        types_found["Separable"] = f
    if is_linear(f):
        types_found["Linear"] = f
    if is_homogeneous(f):
        types_found["Homogeneous"] = f

    if not types_found:
        return "⚠️ ODE type not recognised", {}

    label = "✅ Detected: " + ", ".join(types_found.keys())
    return label, types_found

# =========================================================
# STREAMLIT UI
# =========================================================

st.set_page_config(page_title="ODE Solver", page_icon="🧮", layout="wide")

# ── CSS ──
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&display=swap');

html, body, [class*="css"] { font-family: 'JetBrains Mono', monospace; }

.stApp { background: #1A1F2E; color: #ECF0F1; }

h1 { color: #4F8EF7 !important; letter-spacing: 1px; }

.result-box {
    background: #252B3B;
    border-left: 4px solid #4F8EF7;
    border-radius: 8px;
    padding: 16px 20px;
    margin: 12px 0;
    font-size: 1.05rem;
}
.solution-box {
    background: #1E3A2F;
    border-left: 4px solid #2ECC71;
    border-radius: 8px;
    padding: 16px 20px;
    margin: 12px 0;
    font-size: 1.05rem;
}
.error-box {
    background: #3A1E1E;
    border-left: 4px solid #E74C3C;
    border-radius: 8px;
    padding: 16px 20px;
    margin: 12px 0;
}
.type-badge {
    display: inline-block;
    padding: 4px 14px;
    border-radius: 20px;
    font-size: 0.85rem;
    font-weight: bold;
    margin: 4px;
}
.stTextInput input {
    background: #252B3B !important;
    color: #ECF0F1 !important;
    border: 2px solid #4F8EF7 !important;
    border-radius: 8px !important;
    font-size: 1.1rem !important;
    font-family: 'JetBrains Mono', monospace !important;
}
.stButton button {
    border-radius: 8px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-weight: bold !important;
    border: none !important;
    transition: opacity 0.2s;
}
.stButton button:hover { opacity: 0.85; }

div[data-testid="column"] { padding: 4px 6px; }

.example-row {
    background: #252B3B;
    border-radius: 6px;
    padding: 8px 12px;
    margin: 4px 0;
    font-size: 0.9rem;
    cursor: pointer;
    color: #95A5A6;
}
</style>
""", unsafe_allow_html=True)

# ── Header ──
st.markdown("# 🧮 ODE Solver")
st.markdown("<p style='color:#95A5A6;margin-top:-14px;'>Analyze · Classify · Solve first-order ODEs</p>",
            unsafe_allow_html=True)
st.divider()

# ── Session state ──
if "types_found" not in st.session_state:
    st.session_state.types_found = {}
if "classify_label" not in st.session_state:
    st.session_state.classify_label = ""
if "solution" not in st.session_state:
    st.session_state.solution = ""
if "solution_type" not in st.session_state:
    st.session_state.solution_type = ""
if "eq_input" not in st.session_state:
    st.session_state.eq_input = ""

# ── Examples sidebar ──
EXAMPLES = {
    "Linear":      "dy/dx = -2*y + x",
    "Separable":   "dy/dx = x*y",
    "Bernoulli":   "dy/dx = y + x*y**2",
    "Homogeneous": "dy/dx = (x+y)/x",
    "Exact":       "(2*x*y)dx + (x**2)dy = 0",
}

with st.sidebar:
    st.markdown("### 📚 Examples")
    st.markdown("<p style='color:#95A5A6;font-size:0.85rem;'>Click to load an example</p>",
                unsafe_allow_html=True)
    for kind, eq in EXAMPLES.items():
        if st.button(f"[{kind}]  {eq}", key=f"ex_{kind}", use_container_width=True):
            st.session_state.eq_input = eq
            st.session_state.types_found = {}
            st.session_state.classify_label = ""
            st.session_state.solution = ""
            st.rerun()

    st.divider()
    st.markdown("### 💡 Input Tips")
    tips = [
        "`dy/dx = ...` — standard form",
        "`M dx + N dy = 0` — exact form",
        "Powers: `x**2`, `y**3`",
        "Multiply: `2*x*y`",
        "Analyze first → then Solve",
    ]
    for tip in tips:
        st.markdown(f"• {tip}")

# ── Main input ──
eq_input = st.text_input(
    "Enter your first-order ODE:",
    value=st.session_state.eq_input,
    placeholder="e.g.  dy/dx = x*y   or   (2*x*y)dx + (x**2)dy = 0",
    key="main_input"
)

# ── Analyze + Clear row ──
col_analyze, col_clear, col_space = st.columns([1.4, 1, 4])

with col_analyze:
    analyze_clicked = st.button("🔍  Analyze Type", use_container_width=True, type="primary")

with col_clear:
    if st.button("✖  Clear", use_container_width=True):
        st.session_state.types_found = {}
        st.session_state.classify_label = ""
        st.session_state.solution = ""
        st.session_state.eq_input = ""
        st.rerun()

if analyze_clicked and eq_input.strip():
    label, types = classify_equation(eq_input.strip())
    st.session_state.classify_label = label
    st.session_state.types_found = types
    st.session_state.solution = ""
    st.session_state.eq_input = eq_input.strip()

# ── Analysis result ──
if st.session_state.classify_label:
    st.markdown(f"""<div class="result-box">
        <b>Analysis Result:</b><br>{st.session_state.classify_label}
    </div>""", unsafe_allow_html=True)

# ── Solve buttons (only if types detected) ──
COLORS = {
    "Linear":      "#4F8EF7",
    "Separable":   "#2ECC71",
    "Bernoulli":   "#9B59B6",
    "Homogeneous": "#F39C12",
    "Exact":       "#1ABC9C",
}

types = st.session_state.types_found

if types:
    st.markdown("**Solve as →**  *(only available for detected types)*")
    cols = st.columns(len(types))

    for i, (name, f_or_flag) in enumerate(types.items()):
        with cols[i]:
            color = COLORS.get(name, "#888")
            st.markdown(f"""
            <style>
            div[data-testid="column"]:nth-child({i+1}) button {{
                background-color: {color} !important;
                color: white !important;
            }}
            </style>""", unsafe_allow_html=True)

            if st.button(f"Solve {name}", key=f"solve_{name}", use_container_width=True):
                eq_str = st.session_state.eq_input
                if name == "Exact":
                    res = solve_exact(eq_str)
                elif name == "Linear":
                    res = solve_linear(f_or_flag)
                elif name == "Separable":
                    res = solve_separable(f_or_flag)
                elif name == "Bernoulli":
                    res = solve_bernoulli(f_or_flag)
                elif name == "Homogeneous":
                    res = solve_homogeneous(f_or_flag)
                else:
                    res = "❌ Unknown type"
                st.session_state.solution = res
                st.session_state.solution_type = name

# ── Solution output ──
if st.session_state.solution:
    sol = st.session_state.solution
    stype = st.session_state.solution_type
    color = COLORS.get(stype, "#2ECC71")
    is_error = sol.startswith("❌")
    box_class = "error-box" if is_error else "solution-box"
    icon = "❌" if is_error else "✅"

    st.markdown(f"""<div class="{box_class}">
        <b>{icon} [{stype} Solution]</b><br><br>
        <code style="font-size:1.1rem;color:{'#E74C3C' if is_error else '#2ECC71'};">{sol}</code>
    </div>""", unsafe_allow_html=True)

st.divider()
st.markdown("<p style='color:#555;font-size:0.8rem;text-align:center;'>ODE Solver · SymPy powered</p>",
            unsafe_allow_html=True)
