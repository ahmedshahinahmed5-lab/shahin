import streamlit as st
import sympy as sp
import re

# ─────────────────────────── symbols ────────────────────────────
x, y, t, C1 = sp.symbols('x y t C1')
_DX_ = sp.Symbol('_DX_')
_DY_ = sp.Symbol('_DY_')

# ─────────────────────────── helpers ────────────────────────────

def get_MN(eq_str):
    eq_str = eq_str.replace(" ", "")
    lhs, rhs = eq_str.split("=")
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

def parse_equation(eq_str):
    eq_str = eq_str.replace(" ", "")
    _, rhs = eq_str.split("=", 1)
    return sp.sympify(rhs)

def extract_f(eq_str):
    try:
        return parse_equation(eq_str)
    except:
        return None

# ─────────────────────────── classifiers ────────────────────────

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

# ─────────────────────────── solvers ────────────────────────────

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
        h_inv = sp.simplify(f / g)
        h = sp.simplify(1 / h_inv)
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
        v_sol = (sp.integrate(new_Q * mu, x) + C1) / mu
        y_sol = sp.simplify(v_sol ** (sp.Rational(1, 1 - n)))
        return f"y = {y_sol}   [v = y^{1-n}, n={n}]"
    except Exception as e:
        return f"❌ Could not solve: {e}"

def solve_homogeneous(f):
    try:
        v = sp.Function('v')(x)
        y_sub = v * x
        f_sub = f.subs(y, y_sub)
        F_v = sp.simplify(f_sub)
        rhs = sp.simplify(F_v - v)
        if rhs == 0:
            return "y = C·x"
        sep_lhs = sp.integrate(1 / rhs, v)
        sep_rhs = sp.integrate(1 / x, x)
        return f"{sep_lhs} = {sep_rhs} + C   [v = y/x]"
    except Exception as e:
        return f"❌ Could not solve: {e}"

def solve_exact(eq_str):
    try:
        M, N = get_MN(eq_str)
        F = sp.integrate(M, x)
        g_prime = sp.simplify(N - sp.diff(F, y))
        g = sp.integrate(g_prime, y)
        F_total = sp.simplify(F + g)
        return f"F(x,y) = {F_total} = C"
    except Exception as e:
        return f"❌ Could not solve: {e}"

def solve_nonexact(eq_str):
    try:
        eq_str_clean = eq_str.replace(" ", "")
        M, N = get_MN(eq_str_clean)
        diff = sp.simplify(sp.diff(M, y) - sp.diff(N, x))
        mu_expr = None
        mu_type = ""
        mu_x_expr = sp.simplify(diff / N)
        if not mu_x_expr.has(y):
            mu_expr = sp.exp(sp.integrate(mu_x_expr, x))
            mu_type = "μ(x)"
        else:
            mu_y_expr = sp.simplify(-diff / M)
            if not mu_y_expr.has(x):
                mu_expr = sp.exp(sp.integrate(mu_y_expr, y))
                mu_type = "μ(y)"
        if mu_expr is None:
            return "❌ Could not find integrating factor"
        M2 = sp.expand(mu_expr * M)
        N2 = sp.expand(mu_expr * N)
        F = sp.integrate(M2, x)
        g_prime = sp.simplify(N2 - sp.diff(F, y))
        g = sp.integrate(g_prime, y)
        F_total = sp.simplify(F + g)
        return f"Integrating factor {mu_type} = {sp.simplify(mu_expr)}\nF(x,y) = {F_total} = C"
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
        return (f"Homogeneous sol:  y_h = C·e^(-∫P dx) = {sp.simplify(y_h)}\n"
                f"Particular sol:   y_p = {y_p}\n"
                f"General solution: y = {y_general}")
    except Exception as e:
        return f"❌ Could not solve: {e}"

def classify_equation(eq_str):
    valid, is_ode, msg, eq_std = analyze_input(eq_str)
    if not valid or not is_ode:
        return msg, {}
    types_found = {}
    if is_exact(eq_str):
        types_found['Exact'] = True
    elif is_nonexact(eq_str):
        types_found['NonExact'] = True
    if eq_std and "dy/dx" in eq_std:
        f = extract_f(eq_std)
        if f is None:
            return "❌ Cannot extract dy/dx", {}
        if is_bernoulli(f):
            types_found['Bernoulli'] = f
        if is_separable(f):
            types_found['Separable'] = f
        if is_linear(f):
            types_found['Linear'] = f
        if is_nonhomogeneous_linear(f):
            types_found['NonHomogeneous'] = f
        if is_homogeneous(f):
            types_found['Homogeneous'] = f
    if not types_found:
        return "⚠️ ODE type not recognised", {}
    label = "✅ Type(s): " + ", ".join(types_found.keys())
    return label, types_found

# ─────────────────────────── steps ──────────────────────────────

STEPS = {
    "Linear": """**Steps: Linear Equation**
1. Standard form: `dy/dx + P(x)·y = Q(x)`
2. Integrating factor: `μ = e^(∫P dx)`
3. Multiply both sides: `d/dx(y·μ) = μ·Q`
4. Integrate: `y·μ = ∫μ·Q dx + C`
5. Solve for y: `y = (∫μ·Q dx + C) / μ`""",

    "Separable": """**Steps: Separable Equation**
1. Separable form: `dy/dx = g(x)·h(y)`
2. Separate variables: `dy / h(y) = g(x) dx`
3. Integrate left: `∫ dy/h(y)`
4. Integrate right: `∫ g(x) dx`
5. Add constant C: `∫dy/h(y) = ∫g(x)dx + C`""",

    "Bernoulli": """**Steps: Bernoulli Equation**
1. Identify form: `dy/dx + P(x)y = Q(x)y^n`
2. Substitution: `v = y^(1-n)`
3. Transform to linear: `dv/dx + (1-n)P·v = (1-n)Q`
4. Solve linear ODE for v
5. Back-substitute: `y = v^(1/(1-n))`""",

    "Homogeneous": """**Steps: Homogeneous Equation**
1. Verify: `f(tx, ty) = t^k · f(x, y)`
2. Substitution: `y = v·x`, `dy/dx = v + x·dv/dx`
3. Substitute: `v + x·dv/dx = F(v)`
4. Separate: `dv / (F(v) - v) = dx/x`
5. Integrate & back-substitute: `v = y/x`""",

    "Exact": """**Steps: Exact Equation**
1. Write as: `M(x,y)dx + N(x,y)dy = 0`
2. Check exactness: `∂M/∂y = ∂N/∂x`
3. Find potential: `F = ∫M dx + h(y)`
4. Find h(y) from: `∂F/∂y = N ⟹ h'(y)`
5. Solution: `F(x,y) = C`""",

    "NonExact": """**Steps: Non-Exact Equation**
1. Write as: `M dx + N dy = 0`
2. Verify `∂M/∂y ≠ ∂N/∂x` (not exact)
3. Find integrating factor μ:
   - `(∂M/∂y - ∂N/∂x)/N = f(x)` only → `μ = e^(∫f dx)`
   - `(∂N/∂x - ∂M/∂y)/M = g(y)` only → `μ = e^(∫g dy)`
4. Multiply: `M* = μM`, `N* = μN` (now exact)
5. Solve as exact equation: `F(x,y) = C`""",

    "NonHomogeneous": """**Steps: Non-Homogeneous Linear**
1. Standard form: `dy/dx + P(x)y = Q(x)`
2. Solve homogeneous: `dy/dx + Py = 0` → `y_h = C · e^(-∫P dx)`
3. Variation of parameters: `y_p = u(x) · e^(-∫P dx)`
4. Find u(x): `u'(x) = Q · e^(∫P dx)` → `u = ∫ Q · e^(∫P dx) dx`
5. General solution: `y = y_h + y_p`""",
}

BTN_COLORS = {
    "Linear":         "#386EC6",
    "Separable":      "#2EC46C",
    "Bernoulli":      "#8F5FA1",
    "Homogeneous":    "#D68910",
    "Exact":          "#17A589",
    "NonExact":       "#E74C3C",
    "NonHomogeneous": "#1A5276",
}

EXAMPLES = [
    ("Linear",         "dy/dx + 2*x*y = exp(-x**2)"),
    ("Separable",      "dy/dx = x*y"),
    ("Bernoulli",      "dy/dx + 2*x*y = y**4"),
    ("Homogeneous",    "dy/dx = (x+y)/x"),
    ("Exact",          "(2*x*y)*dx + (x**2)*dy = 0"),
    ("NonExact",       "y*dx + 2*x*dy = 0"),
    ("NonHomogeneous", "dy/dx - y = x**2"),
]

# ─────────────────────────── Streamlit UI ───────────────────────

st.set_page_config(
    page_title="ODE Solver",
    page_icon="📐",
    layout="wide"
)

st.markdown("""
<style>
    body { background-color: #1A1F2E; }
    .main { background-color: #1A1F2E; }
    .stApp { background-color: #1A1F2E; color: #ECF0F1; }
    h1, h2, h3 { color: #4F8EF7; }
    .stTextInput > div > div > input {
        background-color: #2E3650;
        color: #ECF0F1;
        border: 2px solid #4F8EF7;
        font-size: 18px;
        font-family: Consolas, monospace;
    }
    .stButton > button {
        font-family: Consolas, monospace;
        font-weight: bold;
        border-radius: 8px;
    }
    .result-box {
        background-color: #252B3B;
        border-radius: 12px;
        padding: 16px;
        font-family: Consolas, monospace;
        white-space: pre-wrap;
    }
    .steps-box {
        background-color: #2E3650;
        border-left: 4px solid #F39C12;
        border-radius: 8px;
        padding: 14px;
        margin-bottom: 10px;
    }
    .answer-box {
        background-color: #1A3B2E;
        border-left: 4px solid #1ABC9C;
        border-radius: 8px;
        padding: 14px;
        font-size: 16px;
    }
    .example-card {
        background-color: #252B3B;
        border-radius: 10px;
        padding: 10px 14px;
        margin-bottom: 6px;
    }
    .tip-card {
        background-color: #252B3B;
        border-radius: 10px;
        padding: 8px 14px;
        margin-bottom: 6px;
        color: #ECF0F1;
        font-family: Consolas, monospace;
    }
</style>
""", unsafe_allow_html=True)

# ── Title ──
st.markdown("<h1 style='text-align:center; font-family:Consolas;'>📐 Ordinary Differential Equation Solver</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#95A5A6; font-family:Consolas;'>Enter your equation · Analyze · Solve</p>", unsafe_allow_html=True)

st.divider()

# ── Session state ──
if "eq_input" not in st.session_state:
    st.session_state.eq_input = ""
if "types_found" not in st.session_state:
    st.session_state.types_found = {}
if "analysis_msg" not in st.session_state:
    st.session_state.analysis_msg = ""
if "solve_result" not in st.session_state:
    st.session_state.solve_result = None
if "solve_type" not in st.session_state:
    st.session_state.solve_type = None

# ── Input row ──
col_input, col_btns = st.columns([3, 1])

with col_input:
    eq_input = st.text_input(
        "Enter first-order ODE:",
        value=st.session_state.eq_input,
        placeholder="e.g.  dy/dx + 2*x*y = exp(-x**2)",
        key="eq_field"
    )

with col_btns:
    st.write("")
    st.write("")
    c1, c2 = st.columns(2)
    analyze_clicked = c1.button("🔍 Analyze", use_container_width=True)
    clear_clicked   = c2.button("✖ Clear",   use_container_width=True)

if clear_clicked:
    st.session_state.eq_input = ""
    st.session_state.types_found = {}
    st.session_state.analysis_msg = ""
    st.session_state.solve_result = None
    st.session_state.solve_type = None
    st.rerun()

if analyze_clicked and eq_input.strip():
    msg, types = classify_equation(eq_input.strip())
    st.session_state.eq_input = eq_input.strip()
    st.session_state.types_found = types
    st.session_state.analysis_msg = msg
    st.session_state.solve_result = None
    st.session_state.solve_type = None

# ── Analysis result ──
if st.session_state.analysis_msg:
    if "✅" in st.session_state.analysis_msg:
        st.success(st.session_state.analysis_msg)
    elif "⚠️" in st.session_state.analysis_msg:
        st.warning(st.session_state.analysis_msg)
    else:
        st.error(st.session_state.analysis_msg)

# ── Solve buttons ──
if st.session_state.types_found:
    st.markdown("**Solve as →**")
    cols = st.columns(len(BTN_COLORS))
    for i, (name, color) in enumerate(BTN_COLORS.items()):
        disabled = name not in st.session_state.types_found
        label = f"Solve {name}"
        with cols[i]:
            if st.button(label, key=f"solve_{name}", disabled=disabled, use_container_width=True):
                st.session_state.solve_type = name
                eq = st.session_state.eq_input
                f_or_flag = st.session_state.types_found.get(name)
                if name == "Exact":
                    res = solve_exact(eq)
                elif name == "NonExact":
                    res = solve_nonexact(eq)
                elif name == "Linear":
                    res = solve_linear(f_or_flag)
                elif name == "Separable":
                    res = solve_separable(f_or_flag)
                elif name == "Bernoulli":
                    res = solve_bernoulli(f_or_flag)
                elif name == "Homogeneous":
                    res = solve_homogeneous(f_or_flag)
                elif name == "NonHomogeneous":
                    res = solve_nonhomogeneous(f_or_flag)
                else:
                    res = "❌ Unknown type"
                st.session_state.solve_result = res

# ── Steps + Answer display ──
if st.session_state.solve_type and st.session_state.solve_result:
    stype = st.session_state.solve_type
    st.markdown("---")
    st.markdown(f"<div class='steps-box'>{STEPS.get(stype, '')}</div>", unsafe_allow_html=True)
    answer_text = st.session_state.solve_result.replace("\n", "<br>")
    st.markdown(
        f"<div class='answer-box'>► <b>Answer [{stype}]:</b><br>{answer_text}</div>",
        unsafe_allow_html=True
    )

st.divider()

# ── Bottom: Examples + Tips ──
col_ex, col_tip = st.columns(2)

with col_ex:
    st.markdown("### 📚 Example Equations")
    for kind, eq in EXAMPLES:
        color = BTN_COLORS.get(kind, "#4F8EF7")
        col_label, col_eq, col_try = st.columns([1.2, 2.5, 0.8])
        col_label.markdown(f"<span style='color:{color}; font-family:Consolas; font-weight:bold;'>[{kind}]</span>", unsafe_allow_html=True)
        col_eq.markdown(f"<code style='color:#ECF0F1;'>{eq}</code>", unsafe_allow_html=True)
        if col_try.button("Try", key=f"try_{kind}_{eq}"):
            msg, types = classify_equation(eq)
            st.session_state.eq_input = eq
            st.session_state.types_found = types
            st.session_state.analysis_msg = msg
            st.session_state.solve_result = None
            st.session_state.solve_type = None
            st.rerun()

with col_tip:
    st.markdown("### 💡 Input Tips")
    tips = [
        "Use  `dy/dx = ...`  for standard form",
        "Use  `M(x,y)dx + N(x,y)dy = 0`  for exact",
        "Write powers as  `x**2`  or  `y**3`",
        "Use  `*`  for multiplication:  `2*x*y`",
        "Click 🔍 Analyze first, then Solve",
        "Solve buttons light up for matched types",
    ]
    for tip in tips:
        st.markdown(f"<div class='tip-card'>• {tip}</div>", unsafe_allow_html=True)
