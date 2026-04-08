import streamlit as st
import sympy as sp
import re

# =========================================================
# SYMBOLS
# =========================================================
x, y, t, C1 = sp.symbols('x y t C1')
_DX_ = sp.Symbol('_DX_')
_DY_ = sp.Symbol('_DY_')

# =========================================================
# HELPERS
# =========================================================

def get_MN(eq_str):
    eq_str = eq_str.replace(" ", "")
    lhs, rhs = eq_str.split("=", 1)
    lhs2 = re.sub(r'\bdx\b', '_DX_', lhs)
    lhs2 = re.sub(r'\bdy\b', '_DY_', lhs2)
    rhs2 = re.sub(r'\bdx\b', '_DX_', rhs)
    rhs2 = re.sub(r'\bdy\b', '_DY_', rhs2)
    expr = sp.sympify(lhs2) - sp.sympify(rhs2)
    return expr.coeff(_DX_), expr.coeff(_DY_)

def standardize_eq(eq_str):
    eq_str = eq_str.replace(" ", "")
    if "dy/dx" in eq_str:
        return eq_str
    if "dy" in eq_str and "dx" in eq_str and "=" in eq_str:
        try:
            M, N = get_MN(eq_str)
            if M == 0 or N == 0:
                return None
            return f"dy/dx = {sp.simplify(-M / N)}"
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

def extract_f(eq_str):
    try:
        eq_str = eq_str.replace(" ", "")
        _, rhs = eq_str.split("=", 1)
        return sp.sympify(rhs)
    except:
        return None

# =========================================================
# CLASSIFIERS
# =========================================================

def is_separable(f):
    try:
        f = sp.simplify(f)
        fx = f.subs(y, 1)
        fy = sp.simplify(f / fx)
        return not fx.has(y) and not fy.has(x)
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

def is_bernoulli(f):
    try:
        powers = set()
        for term in sp.expand(f).as_ordered_terms():
            if term.has(y):
                try:
                    p = int(sp.Poly(term, y).degree())
                    powers.add(p)
                except:
                    pass
        return len(powers) >= 2 and 1 in powers and any(p not in (0, 1) for p in powers)
    except:
        return False

def is_homogeneous(f):
    try:
        ratio = sp.simplify(f.subs({x: t*x, y: t*y}) / f)
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

def is_nonexact(eq_str):
    try:
        eq_str = eq_str.replace(" ", "")
        if "dx" not in eq_str or "dy" not in eq_str or "=" not in eq_str:
            return False
        M, N = get_MN(eq_str)
        diff = sp.simplify(sp.diff(M, y) - sp.diff(N, x))
        if diff == 0:
            return False
        mu_x = sp.simplify(diff / N)
        if not mu_x.has(y):
            return True
        mu_y = sp.simplify(-diff / M)
        if not mu_y.has(x):
            return True
        return False
    except:
        return False

def is_nonhomogeneous_linear(f):
    try:
        if not is_linear(f):
            return False
        P = -sp.diff(f, y)
        Q = sp.simplify(f + P * y)
        return Q != 0 and Q.has(x)
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
        sol = (sp.integrate(Q * mu, x) + C1) / mu
        return f"y = {sp.simplify(sol)}"
    except Exception as e:
        return f"❌ Could not solve: {e}"

def solve_separable(f):
    try:
        g = f.subs(y, 1)
        h = sp.simplify(1 / (f / g))
        return f"{sp.integrate(h, y)} = {sp.integrate(g, x)} + C"
    except Exception as e:
        return f"❌ Could not solve: {e}"

def solve_bernoulli(f):
    try:
        powers = {}
        for term in sp.expand(f).as_ordered_terms():
            if term.has(y):
                p = int(sp.Poly(term, y).degree())
                powers[p] = powers.get(p, sp.Integer(0)) + sp.simplify(term / y**p)
        n = max(p for p in powers if p not in (0, 1))
        P_c = -powers.get(1, sp.Integer(0))
        Q_c = powers.get(n, sp.Integer(0))
        mu = sp.exp(sp.integrate((1 - n) * P_c, x))
        v_sol = sp.simplify((sp.integrate((1 - n) * Q_c * mu, x) + C1) / mu)
        return f"y = {sp.simplify(v_sol ** sp.Rational(1, 1-n))}   [v = y^{1-n}, n={n}]"
    except Exception as e:
        return f"❌ Could not solve: {e}"

def solve_homogeneous(f):
    try:
        v = sp.Function('v')(x)
        rhs = sp.simplify(f.subs(y, v * x) - v)
        if rhs == 0:
            return "y = C·x"
        return f"{sp.integrate(1/rhs, v)} = {sp.integrate(1/x, x)} + C   [v = y/x]"
    except Exception as e:
        return f"❌ Could not solve: {e}"

def solve_exact(eq_str):
    try:
        M, N = get_MN(eq_str)
        F = sp.integrate(M, x)
        g = sp.integrate(sp.simplify(N - sp.diff(F, y)), y)
        return f"F(x,y) = {sp.simplify(F + g)} = C"
    except Exception as e:
        return f"❌ Could not solve: {e}"

def solve_nonexact(eq_str):
    try:
        eq_str_clean = eq_str.replace(" ", "")
        M, N = get_MN(eq_str_clean)
        diff_val = sp.simplify(sp.diff(M, y) - sp.diff(N, x))
        mu_expr = None
        mu_type = ""
        mu_x_expr = sp.simplify(diff_val / N)
        if not mu_x_expr.has(y):
            mu_expr = sp.exp(sp.integrate(mu_x_expr, x))
            mu_type = "μ(x)"
        else:
            mu_y_expr = sp.simplify(-diff_val / M)
            if not mu_y_expr.has(x):
                mu_expr = sp.exp(sp.integrate(mu_y_expr, y))
                mu_type = "μ(y)"
        if mu_expr is None:
            return "❌ Could not find integrating factor"
        M2 = sp.expand(mu_expr * M)
        N2 = sp.expand(mu_expr * N)
        F = sp.integrate(M2, x)
        g = sp.integrate(sp.simplify(N2 - sp.diff(F, y)), y)
        F_total = sp.simplify(F + g)
        return (f"Integrating factor {mu_type} = {sp.simplify(mu_expr)}\n"
                f"F(x,y) = {F_total} = C")
    except Exception as e:
        return f"❌ Could not solve: {e}"

def solve_nonhomogeneous(f):
    try:
        P = -sp.diff(f, y)
        Q = sp.simplify(f + P * y)
        mu = sp.exp(sp.integrate(P, x))
        y_h = C1 / mu
        y_p = sp.simplify(sp.integrate(Q * mu, x) / mu)
        y_general = sp.simplify(y_h + y_p)
        return (f"Homogeneous sol:  y_h = {sp.simplify(y_h)}\n"
                f"Particular sol:   y_p = {y_p}\n"
                f"General solution: y = {y_general}")
    except Exception as e:
        return f"❌ Could not solve: {e}"

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
    elif is_nonexact(eq_str):
        types_found["NonExact"] = True

    if eq_std and "dy/dx" in eq_std:
        f = extract_f(eq_std)
        if f is None:
            return "❌ Cannot extract dy/dx", {}
        if is_bernoulli(f):       types_found["Bernoulli"]       = f
        if is_separable(f):       types_found["Separable"]       = f
        if is_linear(f):          types_found["Linear"]          = f
        if is_nonhomogeneous_linear(f): types_found["NonHomogeneous"] = f
        if is_homogeneous(f):     types_found["Homogeneous"]     = f

    if not types_found:
        return "⚠️ ODE type not recognised", {}

    return "✅ Type(s): " + ", ".join(types_found.keys()), types_found

# =========================================================
# STEPS
# =========================================================

STEPS = {
    "Linear": [
        ("Standard Form",      "Rewrite as:  **dy/dx + P(x)·y = Q(x)**"),
        ("Integrating Factor",  "Compute:  **μ(x) = e^(∫P(x) dx)**"),
        ("Multiply Both Sides", "Equation becomes:  **d/dx(y·μ) = μ·Q(x)**"),
        ("Integrate",           "Both sides:  **y·μ = ∫μ·Q(x) dx + C**"),
        ("Solve for y",         "Divide by μ:  **y = (∫μ·Q dx + C) / μ**"),
    ],
    "Separable": [
        ("Separable Form",      "Rewrite as:  **dy/dx = g(x)·h(y)**"),
        ("Separate Variables",  "Rearrange:  **dy / h(y) = g(x) dx**"),
        ("Integrate Left Side", "Compute:  **∫ dy/h(y)**"),
        ("Integrate Right Side","Compute:  **∫ g(x) dx**"),
        ("Add Constant",        "Combine:  **∫dy/h(y) = ∫g(x)dx + C**"),
    ],
    "Bernoulli": [
        ("Identify Form",       "Equation is:  **dy/dx + P(x)y = Q(x)yⁿ**"),
        ("Substitution",        "Let:  **v = y^(1−n)**"),
        ("Transform to Linear", "New ODE:  **dv/dx + (1−n)P(x)v = (1−n)Q(x)**"),
        ("Solve Linear ODE",    "Apply integrating factor method to find **v(x)**"),
        ("Back-Substitute",     "Recover y:  **y = v^(1/(1−n))**"),
    ],
    "Homogeneous": [
        ("Verify Homogeneity",  "Check:  **f(tx, ty) = tᵏ f(x, y)**"),
        ("Substitution",        "Let:  **y = v·x**,  so  **dy/dx = v + x·dv/dx**"),
        ("Substitute",          "Equation becomes:  **v + x·dv/dx = F(v)**"),
        ("Separate & Integrate","Rearrange:  **dv / (F(v)−v) = dx/x**,  then integrate"),
        ("Back-Substitute",     "Replace v with:  **v = y/x**"),
    ],
    "Exact": [
        ("Standard Form",       "Write as:  **M(x,y)dx + N(x,y)dy = 0**"),
        ("Check Exactness",     "Verify:  **∂M/∂y = ∂N/∂x**"),
        ("Find Potential F",    "Integrate:  **F = ∫M dx + h(y)**"),
        ("Find h(y)",           "Differentiate F w.r.t. y and set equal to N to find **h'(y)**"),
        ("Write Solution",      "Final answer:  **F(x, y) = C**"),
    ],
    "NonExact": [
        ("Standard Form",        "Write as:  **M dx + N dy = 0**"),
        ("Verify Not Exact",     "Confirm:  **∂M/∂y ≠ ∂N/∂x**"),
        ("Find Integrating Factor", "Try **(∂M/∂y − ∂N/∂x)/N = f(x)** → μ = e^(∫f dx)\nOr **(∂N/∂x − ∂M/∂y)/M = g(y)** → μ = e^(∫g dy)"),
        ("Multiply",             "Set  **M* = μM**,  **N* = μN**  — now exact"),
        ("Solve as Exact",       "Apply exact equation method:  **F(x,y) = C**"),
    ],
    "NonHomogeneous": [
        ("Standard Form",        "Write as:  **dy/dx + P(x)y = Q(x)**,  Q(x) ≠ 0"),
        ("Solve Homogeneous Part","Set Q=0:  **y_h = C · e^(−∫P dx)**"),
        ("Variation of Parameters","Let:  **y_p = u(x) · e^(−∫P dx)**"),
        ("Find u(x)",            "Compute:  **u'(x) = Q · e^(∫P dx)**  →  integrate for u"),
        ("General Solution",     "Combine:  **y = y_h + y_p**"),
    ],
}

BTN_COLORS = {
    "Linear":          "#386EC6",
    "Separable":       "#2EC46C",
    "Bernoulli":       "#8F5FA1",
    "Homogeneous":     "#D68910",
    "Exact":           "#17A589",
    "NonExact":        "#E74C3C",
    "NonHomogeneous":  "#1A5276",
}

EXAMPLES = [
    ("Linear",          "dy/dx + 2*x*y = exp(-x**2)"),
    ("Separable",       "dy/dx = x*y"),
    ("Bernoulli",       "dy/dx + 2*x*y = y**4"),
    ("Homogeneous",     "dy/dx = (x+y)/x"),
    ("Exact",           "(2*x*y)*dx + (x**2)*dy = 0"),
    ("NonExact",        "y*dx + 2*x*dy = 0"),
    ("NonHomogeneous",  "dy/dx - y = x**2"),
]

# =========================================================
# UI
# =========================================================

st.set_page_config(page_title="ODE Solver", page_icon="📐", layout="wide")

st.markdown("""
<style>
    body, .main, .stApp { background-color: #1A1F2E; color: #ECF0F1; }
    h1, h2, h3 { color: #4F8EF7; }
    .stTextInput > div > div > input {
        background-color: #2E3650;
        color: #ECF0F1;
        border: 2px solid #4F8EF7;
        font-size: 17px;
        font-family: Consolas, monospace;
    }
    .stButton > button {
        font-family: Consolas, monospace;
        font-weight: bold;
        border-radius: 8px;
    }
    .step-box {
        background: #1e2a3a;
        border-left: 4px solid #4F8EF7;
        border-radius: 6px;
        padding: 10px 16px;
        margin: 6px 0;
    }
    .step-num   { color: #4F8EF7; font-weight: bold; font-size: 0.82em; text-transform: uppercase; letter-spacing: 1px; }
    .step-title { color: #F39C12; font-weight: bold; margin-bottom: 2px; }
    .step-body  { color: #ECF0F1; white-space: pre-line; }
    .answer-box {
        background-color: #1A3B2E;
        border-left: 4px solid #1ABC9C;
        border-radius: 8px;
        padding: 16px;
        font-size: 15px;
        font-family: Consolas, monospace;
        white-space: pre-wrap;
    }
    .tip-card {
        background-color: #252B3B;
        border-radius: 10px;
        padding: 8px 14px;
        margin-bottom: 6px;
        color: #ECF0F1;
        font-family: Consolas, monospace;
    }
    [data-testid="stSidebar"] { background-color: #141926; }
</style>
""", unsafe_allow_html=True)

# ── Title ──
st.markdown("<h1 style='text-align:center; font-family:Consolas;'>📐 ODE Solver</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#95A5A6; font-family:Consolas;'>Enter your equation · Analyze · Solve</p>", unsafe_allow_html=True)
st.divider()

# ── Session state ──
for k, default in [("eq_input",""), ("types_found",{}), ("analysis_msg",""),
                   ("solve_result",None), ("solve_type",None)]:
    if k not in st.session_state:
        st.session_state[k] = default

# ── Sidebar: Examples + Tips ──
with st.sidebar:
    st.markdown("## 📚 Examples")
    for kind, eq in EXAMPLES:
        color = BTN_COLORS.get(kind, "#4F8EF7")
        label_html = f"<span style='color:{color};font-weight:bold;'>[{kind}]</span>"
        col_l, col_b = st.columns([2.5, 1])
        col_l.markdown(f"{label_html}<br><code style='color:#BDC3C7;font-size:0.78em;'>{eq}</code>", unsafe_allow_html=True)
        if col_b.button("Try", key=f"try_{kind}_{eq}"):
            msg, types = classify_equation(eq)
            st.session_state.eq_input     = eq
            st.session_state.types_found  = types
            st.session_state.analysis_msg = msg
            st.session_state.solve_result = None
            st.session_state.solve_type   = None
            st.rerun()

    st.divider()
    st.markdown("## 💡 Input Tips")
    tips = [
        "Use `dy/dx = ...` for standard form",
        "Use `M dx + N dy = 0` for exact/non-exact",
        "Powers: `x**2`, `y**3`",
        "Multiply: `2*x*y`",
        "Click 🔍 Analyze first, then Solve",
    ]
    for tip in tips:
        st.markdown(f"<div class='tip-card'>• {tip}</div>", unsafe_allow_html=True)

# ── Input row ──
col_input, col_btns = st.columns([3, 1])

with col_input:
    eq_input = st.text_input(
        "ODE:",
        value=st.session_state.eq_input,
        placeholder="e.g.  dy/dx + 2*x*y = exp(-x**2)  or  y*dx + 2*x*dy = 0",
        label_visibility="collapsed"
    )
    st.session_state.eq_input = eq_input

with col_btns:
    st.write("")
    st.write("")
    c1, c2 = st.columns(2)
    analyze_clicked = c1.button("🔍 Analyze", use_container_width=True)
    clear_clicked   = c2.button("✖ Clear",    use_container_width=True)

if clear_clicked:
    for k, default in [("eq_input",""), ("types_found",{}), ("analysis_msg",""),
                       ("solve_result",None), ("solve_type",None)]:
        st.session_state[k] = default
    st.rerun()

if analyze_clicked and eq_input.strip():
    msg, types = classify_equation(eq_input.strip())
    st.session_state.eq_input     = eq_input.strip()
    st.session_state.types_found  = types
    st.session_state.analysis_msg = msg
    st.session_state.solve_result = None
    st.session_state.solve_type   = None

# ── Analysis result ──
if st.session_state.analysis_msg:
    msg = st.session_state.analysis_msg
    if "✅" in msg:   st.success(msg)
    elif "⚠️" in msg: st.warning(msg)
    else:              st.error(msg)

# ── Solve buttons ──
if st.session_state.types_found:
    st.markdown("**Solve as →**")
    cols = st.columns(len(BTN_COLORS))
    for i, (name, color) in enumerate(BTN_COLORS.items()):
        disabled = name not in st.session_state.types_found
        with cols[i]:
            if st.button(f"Solve {name}", key=f"solve_{name}",
                         disabled=disabled, use_container_width=True):
                st.session_state.solve_type = name
                eq    = st.session_state.eq_input
                f_val = st.session_state.types_found.get(name)
                solvers = {
                    "Exact":          lambda: solve_exact(eq),
                    "NonExact":       lambda: solve_nonexact(eq),
                    "Linear":         lambda: solve_linear(f_val),
                    "Separable":      lambda: solve_separable(f_val),
                    "Bernoulli":      lambda: solve_bernoulli(f_val),
                    "Homogeneous":    lambda: solve_homogeneous(f_val),
                    "NonHomogeneous": lambda: solve_nonhomogeneous(f_val),
                }
                st.session_state.solve_result = solvers.get(name, lambda: "❌ Unknown")()

# ── Steps + Answer ──
if st.session_state.solve_type and st.session_state.solve_result:
    stype = st.session_state.solve_type
    sol   = st.session_state.solve_result
    st.markdown("---")

    col_steps, col_ans = st.columns([1, 1])

    with col_steps:
        st.subheader(f"📋 Steps — {stype}")
        for i, (title, body) in enumerate(STEPS.get(stype, []), 1):
            st.markdown(f"""
<div class="step-box">
  <div class="step-num">Step {i}</div>
  <div class="step-title">{title}</div>
  <div class="step-body">{body}</div>
</div>""", unsafe_allow_html=True)

    with col_ans:
        st.subheader("✅ Answer")
        if sol.startswith("❌"):
            st.error(sol)
        else:
            st.markdown(f"<div class='answer-box'>► <b>[{stype}]</b><br><br>{sol.replace(chr(10), '<br>')}</div>",
                        unsafe_allow_html=True)
