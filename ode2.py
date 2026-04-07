import streamlit as st
import sympy as sp
import re

x, y, t, C1 = sp.symbols('x y t C1')
_DX_ = sp.Symbol('_DX_')
_DY_ = sp.Symbol('_DY_')

# ─────────────────────────── helpers ────────────────────────────

def get_MN(eq_str):
    """Return (M, N) from M dx + N dy = 0 using safe regex parsing."""
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
    """Parse dy/dx = f(x,y) and return f as sympy expression."""
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
        if not fx.has(y) and not fy.has(x):
            return True
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
    """Not exact but has an integrating factor μ(x) or μ(y)."""
    try:
        eq_str = eq_str.replace(" ", "")
        if "dx" not in eq_str or "dy" not in eq_str or "=" not in eq_str:
            return False
        M, N = get_MN(eq_str)
        diff = sp.simplify(sp.diff(M, y) - sp.diff(N, x))
        if diff == 0:
            return False   # already exact
        # check (My - Nx)/N depends only on x
        mu_x = sp.simplify(diff / N)
        if not mu_x.has(y):
            return True
        # check (Nx - My)/M depends only on y
        mu_y = sp.simplify(-diff / M)
        if not mu_y.has(x):
            return True
        return False
    except:
        return False

def is_nonhomogeneous_linear(f):
    """Linear ODE dy/dx + P(x)y = Q(x) where Q != 0."""
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
    """dy/dx = P(x)*y + Q(x)  →  y = e^(-∫P) [∫Q e^∫P dx + C]"""
    try:
        P = -sp.diff(f, y)          # coefficient of y  (f = -P*y + Q)
        Q = sp.simplify(f + P * y)  # f = -P y + Q  →  Q = f + P y
        mu = sp.exp(sp.integrate(P, x))
        sol = (sp.integrate(Q * mu, x) + C1) / mu
        return f"y = {sp.simplify(sol)}"
    except Exception as e:
        return f"❌ Could not solve: {e}"

def solve_separable(f):
    """dy/dx = g(x)/h(y)  →  ∫h dy = ∫g dx"""
    try:
        g = f.subs(y, 1)        # g(x) part
        h_inv = sp.simplify(f / g)   # 1/h(y) part
        h = sp.simplify(1 / h_inv)   # h(y)
        lhs = sp.integrate(h, y)
        rhs = sp.integrate(g, x)
        return f"{lhs} = {rhs} + C"
    except Exception as e:
        return f"❌ Could not solve: {e}"

def solve_bernoulli(f):
    """y' + P y = Q y^n  →  substitute v = y^(1-n)"""
    try:
        terms = sp.expand(f).as_ordered_terms()
        # find powers of y
        powers = {}
        for term in terms:
            if term.has(y):
                p = sp.degree(term, y)
                powers[p] = powers.get(p, sp.Integer(0)) + sp.simplify(term / y**p)
        n = max(p for p in powers if p not in (0, 1))
        P_coeff = -powers.get(1, sp.Integer(0))
        Q_coeff = powers.get(n, sp.Integer(0))
        v = sp.Function('v')(x)
        # v' + (1-n)P v = (1-n)Q
        new_P = (1 - n) * P_coeff
        new_Q = (1 - n) * Q_coeff
        mu = sp.exp(sp.integrate(new_P, x))
        v_sol = (sp.integrate(new_Q * mu, x) + C1) / mu
        y_sol = sp.simplify(v_sol ** (sp.Rational(1, 1 - n)))
        return f"y = {y_sol}   [v = y^{1-n}, n={n}]"
    except Exception as e:
        return f"❌ Could not solve: {e}"

def solve_homogeneous(f):
    """y' = F(y/x)  →  substitute v = y/x"""
    try:
        v = sp.Function('v')(x)
        y_sub = v * x
        f_sub = f.subs(y, y_sub)
        # dy/dx = v + x dv/dx
        # v + x v' = F(v)  →  x v' = F(v) - v
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
    """M dx + N dy = 0  →  find F(x,y) = C"""
    try:
        M, N = get_MN(eq_str)
        # F_x = M  →  F = ∫M dx + g(y)
        F = sp.integrate(M, x)
        g_prime = sp.simplify(N - sp.diff(F, y))
        # g_prime should depend on y only
        g = sp.integrate(g_prime, y)
        F_total = sp.simplify(F + g)
        return f"F(x,y) = {F_total} = C"
    except Exception as e:
        return f"❌ Could not solve: {e}"

def solve_nonexact(eq_str):
    """Find integrating factor μ(x) or μ(y), multiply, then solve as exact."""
    try:
        eq_str_clean = eq_str.replace(" ", "")
        M, N = get_MN(eq_str_clean)
        diff = sp.simplify(sp.diff(M, y) - sp.diff(N, x))

        mu_expr = None
        mu_type = ""

        # try μ(x)
        mu_x_expr = sp.simplify(diff / N)
        if not mu_x_expr.has(y):
            mu_expr = sp.exp(sp.integrate(mu_x_expr, x))
            mu_type = "μ(x)"
        else:
            # try μ(y)
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
    """dy/dx + P(x)y = Q(x)  solved by variation of parameters: y = y_h + y_p"""
    try:
        P = -sp.diff(f, y)
        Q = sp.simplify(f + P * y)

        # homogeneous solution: y_h = C * e^(-∫P dx)
        mu = sp.exp(sp.integrate(P, x))
        y_h = C1 / mu

        # particular solution: y_p = (1/μ) ∫ Q·μ dx
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

    # For dy/dx form equations, extract f and classify
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

# ─────────────────────────── UI with Streamlit ─────────────────────────────────

# Page config
st.set_page_config(
    page_title="ODE Solver",
    page_icon="📚",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .stButton > button {
        font-family: 'Consolas', monospace;
        font-weight: bold;
    }
    .stTextInput > div > div > input {
        font-family: 'Consolas', monospace;
        font-size: 18px;
    }
    .stTextArea textarea {
        font-family: 'Consolas', monospace;
    }
    .success-text {
        color: #2ECC71;
        font-family: 'Consolas', monospace;
    }
    .info-text {
        color: #4F8EF7;
        font-family: 'Consolas', monospace;
    }
    .warning-text {
        color: #F39C12;
        font-family: 'Consolas', monospace;
    }
    .steps-text {
        color: #F39C12;
        font-family: 'Consolas', monospace;
        background-color: #2E3650;
        padding: 10px;
        border-radius: 5px;
    }
    .answer-text {
        color: #1ABC9C;
        font-family: 'Consolas', monospace;
        font-size: 16px;
        font-weight: bold;
        background-color: #2E3650;
        padding: 10px;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown("<h1 style='text-align: center; color: #4F8EF7;'>📚 Ordinary Differential Equation Solver</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #95A5A6;'>Enter your equation · Analyze · Solve</p>", unsafe_allow_html=True)

# Store current types in session state
if 'current_types' not in st.session_state:
    st.session_state.current_types = {}
if 'current_eq' not in st.session_state:
    st.session_state.current_eq = ""

# Input section
with st.container():
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        eq_input = st.text_input(
            "Enter first-order ODE:",
            value=st.session_state.current_eq,
            placeholder="Example: dy/dx = x*y  or  (2*x*y)*dx + (x**2)*dy = 0",
            key="equation_input"
        )
        
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            analyze_btn = st.button("🔍 Analyze", use_container_width=True, type="primary")
        with col_btn2:
            clear_btn = st.button("✖ Clear", use_container_width=True)

# Analysis result
if clear_btn:
    st.session_state.current_eq = ""
    st.session_state.current_types = {}
    st.rerun()

if analyze_btn and eq_input:
    st.session_state.current_eq = eq_input
    label, types = classify_equation(eq_input)
    st.session_state.current_types = types
    
    if "⚠️" in label or "❌" in label:
        st.warning(label)
    else:
        st.success(label)

# Display current equation info
if st.session_state.current_eq:
    st.markdown(f"<p class='info-text'>📝 Current equation: <b>{st.session_state.current_eq}</b></p>", unsafe_allow_html=True)

# STEPS dictionary
STEPS = {
    "Linear": """```
── Steps: Linear Equation ─────────────────────────────
 Step 1 │ Standard form:      dy/dx + P(x)·y = Q(x)
 Step 2 │ Integrating factor: μ = e^(∫P dx)
 Step 3 │ Multiply both sides: d/dx(y·μ) = μ·Q
 Step 4 │ Integrate:          y·μ = ∫μ·Q dx + C
 Step 5 │ Solve for y:        y = (∫μ·Q dx + C) / μ
───────────────────────────────────────────────────────```""",

    "Separable": """```
── Steps: Separable Equation ──────────────────────────
 Step 1 │ Separable form:     dy/dx = g(x)·h(y)
 Step 2 │ Separate variables: dy / h(y) = g(x) dx
 Step 3 │ Integrate left:     ∫ dy/h(y)
 Step 4 │ Integrate right:    ∫ g(x) dx
 Step 5 │ Add constant C:     ∫dy/h(y) = ∫g(x)dx + C
───────────────────────────────────────────────────────```""",

    "Bernoulli": """```
── Steps: Bernoulli Equation ──────────────────────────
 Step 1 │ Identify form:      dy/dx + P(x)y = Q(x)y^n
 Step 2 │ Substitution:       v = y^(1-n)
 Step 3 │ Transform to linear: dv/dx + (1-n)P*v = (1-n)Q
 Step 4 │ Solve linear ODE for v
 Step 5 │ Back-substitute:    y = v^(1/(1-n))
───────────────────────────────────────────────────────```""",

    "Homogeneous": """```
── Steps: Homogeneous Equation ────────────────────────
 Step 1 │ Verify:             f(tx,ty) = t^k * f(x,y)
 Step 2 │ Substitution:       y = v*x,  dy/dx = v + x*dv/dx
 Step 3 │ Substitute:         v + x*dv/dx = F(v)
 Step 4 │ Separate:           dv / (F(v)-v) = dx/x
 Step 5 │ Integrate & back-sub: v = y/x
───────────────────────────────────────────────────────```""",

    "Exact": """```
── Steps: Exact Equation ──────────────────────────────
 Step 1 │ Write as:           M(x,y)dx + N(x,y)dy = 0
 Step 2 │ Check exactness:    dM/dy = dN/dx
 Step 3 │ Find potential:     F = ∫M dx + h(y)
 Step 4 │ Find h(y) from:     dF/dy = N  =>  h'(y)
 Step 5 │ Solution:           F(x,y) = C
───────────────────────────────────────────────────────```""",

    "NonExact": """```
── Steps: Non-Exact Equation ──────────────────────────
 Step 1 │ Write as:           M dx + N dy = 0
 Step 2 │ Verify dM/dy ≠ dN/dx  (not exact)
 Step 3 │ Find integrating factor μ:
         │  (dM/dy - dN/dx)/N = f(x) only → μ = e^(∫f dx)
         │  (dN/dx - dM/dy)/M = g(y) only → μ = e^(∫g dy)
 Step 4 │ Multiply: M* = μM,  N* = μN  (now exact)
 Step 5 │ Solve as exact equation: F(x,y) = C
───────────────────────────────────────────────────────```""",

    "NonHomogeneous": """```
── Steps: Non-Homogeneous Linear ──────────────────────
 Step 1 │ Standard form:      dy/dx + P(x)y = Q(x)
 Step 2 │ Solve homogeneous:  dy/dx + Py = 0
         │   → y_h = C · e^(-∫P dx)
 Step 3 │ Variation of parameters: y_p = u(x) · e^(-∫P dx)
 Step 4 │ Find u(x):  u'(x) = Q · e^(∫P dx)
         │   → u = ∫ Q · e^(∫P dx) dx
 Step 5 │ General solution:   y = y_h + y_p
───────────────────────────────────────────────────────```""",
}

# Solve buttons section
if st.session_state.current_types:
    st.markdown("---")
    st.markdown("<h3 style='text-align: center;'>🔧 Solve as:</h3>", unsafe_allow_html=True)
    
    # Create columns for buttons dynamically
    type_list = list(st.session_state.current_types.keys())
    cols = st.columns(min(len(type_list), 4))
    
    for idx, ode_type in enumerate(type_list):
        with cols[idx % 4]:
            if st.button(f"Solve {ode_type}", use_container_width=True):
                f_or_flag = st.session_state.current_types.get(ode_type)
                
                # Show steps
                with st.expander("📖 Solution Steps", expanded=True):
                    st.markdown(STEPS.get(ode_type, ""))
                
                # Solve
                with st.spinner(f"Solving {ode_type} equation..."):
                    if ode_type == "Exact":
                        res = solve_exact(st.session_state.current_eq)
                    elif ode_type == "NonExact":
                        res = solve_nonexact(st.session_state.current_eq)
                    elif ode_type == "Linear":
                        res = solve_linear(f_or_flag)
                    elif ode_type == "Separable":
                        res = solve_separable(f_or_flag)
                    elif ode_type == "Bernoulli":
                        res = solve_bernoulli(f_or_flag)
                    elif ode_type == "Homogeneous":
                        res = solve_homogeneous(f_or_flag)
                    elif ode_type == "NonHomogeneous":
                        res = solve_nonhomogeneous(f_or_flag)
                    else:
                        res = "❌ Unknown type"
                
                st.markdown(f"<div class='answer-text'>🎯 Answer [{ode_type}]:<br>{res}</div>", unsafe_allow_html=True)

# Examples and Tips section
st.markdown("---")
col_ex, col_tip = st.columns(2)

with col_ex:
    st.markdown("<h3 style='color: #4F8EF7;'>📚 Example Equations</h3>", unsafe_allow_html=True)
    
    examples = [
        ("Linear", "dy/dx + 2*x*y = exp(-x**2)"),
        ("Separable", "dy/dx = x*y"),
        ("Bernoulli", "dy/dx + 2*x*y = y**4"),
        ("Homogeneous", "dy/dx = (x+y)/x"),
        ("Exact", "(2*x*y)*dx + (x**2)*dy = 0"),
        ("NonExact", "y*dx + 2*x*dy = 0"),
        ("NonHomogeneous", "dy/dx - y = x**2"),
    ]
    
    for kind, eq in examples:
        col1, col2, col3 = st.columns([1, 3, 1])
        with col1:
            st.markdown(f"<span style='color: #4F8EF7;'>[{kind}]</span>", unsafe_allow_html=True)
        with col2:
            st.code(eq, language="python")
        with col3:
            if st.button("Try", key=f"try_{kind}"):
                st.session_state.current_eq = eq
                st.rerun()

with col_tip:
    st.markdown("<h3 style='color: #F39C12;'>💡 Input Tips</h3>", unsafe_allow_html=True)
    
    tips = [
        "Use  dy/dx = ...  for standard form",
        "Use  M(x,y)dx + N(x,y)dy = 0  for exact",
        "Write powers as  x**2  or  y**3",
        "Use  *  for multiplication:  2*x*y",
        "Click 🔍 Analyze first, then Solve",
        "Solve buttons appear for matched types",
    ]
    
    for tip in tips:
        st.markdown(f"• {tip}")

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center; color: #95A5A6;'>ODE Solver • Powered by SymPy & Streamlit</p>", unsafe_allow_html=True)
