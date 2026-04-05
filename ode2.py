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
# SOLVERS WITH STEPS
# =========================================================

def solve_linear_with_steps(f):
    try:
        steps = []
        P = -sp.diff(f, y)
        Q = sp.simplify(f + P * y)
        mu = sp.exp(sp.integrate(P, x))
        integral = sp.integrate(mu * Q, x)
        y_sol = sp.simplify((integral + C1) / mu)
        
        steps = [
            "**Step 1:** Write in standard form: `dy/dx + P(x)y = Q(x)`",
            f"`P(x) = {sp.latex(P)}` , `Q(x) = {sp.latex(Q)}`",
            "**Step 2:** Find integrating factor: `μ = e^(∫P dx)`",
            f"`μ = {sp.latex(mu)}`",
            "**Step 3:** Multiply both sides: `d/dx(y·μ) = μ·Q`",
            "**Step 4:** Integrate: `y·μ = ∫μ·Q dx + C`",
            f"`y·μ = {sp.latex(integral)} + C`",
            "**Step 5:** Solve for y",
            f"`y = {sp.latex(y_sol)}`"
        ]
        return y_sol, steps
    except Exception as e:
        return None, [f"❌ Error: {e}"]

def solve_separable_with_steps(f):
    try:
        steps = []
        g = f.subs(y, 1)
        h = sp.simplify(1 / (f / g))
        left_int = sp.integrate(h, y)
        right_int = sp.integrate(g, x)
        
        steps = [
            "**Step 1:** Write in separable form: `dy/dx = g(x)h(y)`",
            f"`dy/dx = {sp.latex(f)}`",
            "**Step 2:** Separate variables: `dy/h(y) = g(x) dx`",
            f"`dy/{sp.latex(h)} = {sp.latex(g)} dx`",
            "**Step 3:** Integrate both sides",
            f"`∫ {sp.latex(h)} dy = ∫ {sp.latex(g)} dx`",
            f"`{sp.latex(left_int)} = {sp.latex(right_int)} + C`"
        ]
        return sp.Eq(left_int, right_int + C1), steps
    except Exception as e:
        return None, [f"❌ Error: {e}"]

def solve_bernoulli_with_steps(f):
    try:
        steps = []
        powers = {}
        for term in sp.expand(f).as_ordered_terms():
            if term.has(y):
                p = sp.degree(term, y)
                powers[p] = powers.get(p, sp.Integer(0)) + sp.simplify(term / y**p)
        
        n = max(p for p in powers if p not in (0, 1))
        P_c = -powers.get(1, sp.Integer(0))
        Q_c = powers.get(n, sp.Integer(0))
        coeff = 1 - n
        mu = sp.exp(sp.integrate(coeff * P_c, x))
        v_sol = sp.simplify((sp.integrate(coeff * Q_c * mu, x) + C1) / mu)
        y_sol = sp.simplify(v_sol ** sp.Rational(1, 1-n))
        
        steps = [
            "**Step 1:** Identify Bernoulli form: `dy/dx + P(x)y = Q(x)y^n`",
            f"`n = {n}`",
            "**Step 2:** Find P(x) and Q(x)",
            f"`P(x) = {sp.latex(P_c)}` , `Q(x) = {sp.latex(Q_c)}`",
            f"**Step 3:** Substitution `v = y^{{{1-n}}}`",
            f"`dv/dx + ({sp.latex(coeff)})P(x)v = ({sp.latex(coeff)})Q(x)`",
            "**Step 4:** Solve linear equation for v",
            f"`v = {sp.latex(v_sol)}`",
            f"**Step 5:** Substitute back `y = v^{{{1/(1-n)}}}`",
            f"`y = {sp.latex(y_sol)}`"
        ]
        return y_sol, steps
    except Exception as e:
        return None, [f"❌ Error: {e}"]

def solve_homogeneous_with_steps(f):
    try:
        steps = []
        v = sp.Function('v')(x)
        rhs = sp.simplify(f.subs(y, v * x) - v)
        
        steps = [
            "**Step 1:** Check homogeneity: `f(tx,ty) = t^k f(x,y)` ✓",
            "**Step 2:** Substitution `y = vx` , `dy/dx = v + x dv/dx`",
            "**Step 3:** Substitute into equation",
            f"`x dv/dx = {sp.latex(rhs)}`",
            "**Step 4:** Separate variables",
            f"`dv/{sp.latex(rhs)} = dx/x`",
            "**Step 5:** Integrate both sides",
            f"`∫ dv/{sp.latex(rhs)} = ∫ dx/x + C`",
            "**Step 6:** Substitute back `v = y/x`"
        ]
        return None, steps
    except Exception as e:
        return None, [f"❌ Error: {e}"]

def solve_exact_with_steps(eq_str):
    try:
        steps = []
        M, N = get_MN(eq_str)
        dM_dy = sp.diff(M, y)
        dN_dx = sp.diff(N, x)
        F = sp.integrate(M, x)
        dh_dy = sp.simplify(N - sp.diff(F, y))
        h = sp.integrate(dh_dy, y)
        F_final = sp.simplify(F + h)
        
        steps = [
            "**Step 1:** Check exactness: `∂M/∂y = ∂N/∂x`",
            f"`M = {sp.latex(M)}` , `N = {sp.latex(N)}`",
            f"`∂M/∂y = {sp.latex(dM_dy)}` , `∂N/∂x = {sp.latex(dN_dx)}`",
            "✓ Equation is exact",
            "**Step 2:** Find `F = ∫M dx + h(y)`",
            f"`F = {sp.latex(F)} + h(y)`",
            "**Step 3:** Find h(y) from `∂F/∂y = N`",
            f"`dh/dy = {sp.latex(dh_dy)}`",
            f"`h(y) = {sp.latex(h)}`",
            "**Step 4:** General solution: `F(x,y) = C`",
            f"`{sp.latex(F_final)} = C`"
        ]
        return F_final, steps
    except Exception as e:
        return None, [f"❌ Error: {e}"]

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
for key in ("types_found", "label", "solution", "steps", "sol_type", "eq_saved", "selected_method"):
    if key not in st.session_state:
        st.session_state[key] = {} if key == "types_found" else [] if key == "steps" else ""

# ── Examples in Sidebar ──
with st.sidebar:
    st.header("📚 Examples")
    examples = {
        "Linear":       "dy/dx = -2*y + x",
        "Separable":    "dy/dx = x*y",
        "Bernoulli":    "dy/dx = y + x*y**2",
        "Homogeneous":  "dy/dx = (x+y)/x",
        "Exact":        "(2*x*y)dx + (x**2)dy = 0",
    }
    for kind, eq in examples.items():
        if st.button(f"{kind}: {eq}", use_container_width=True):
            st.session_state.eq_saved = eq
            st.session_state.types_found = {}
            st.session_state.label = ""
            st.session_state.solution = ""
            st.session_state.steps = []
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
            st.session_state.steps = []
            st.session_state.eq_saved = eq_input.strip()
            # تعيين أول طريقة كـ selected_method
            if types:
                st.session_state.selected_method = list(types.keys())[0]

with col2:
    if st.button("🗑 Clear", use_container_width=True):
        for key in ("types_found", "label", "solution", "steps", "sol_type", "eq_saved", "selected_method"):
            st.session_state[key] = {} if key == "types_found" else [] if key == "steps" else ""
        st.rerun()

# ── Analysis result ──
if st.session_state.label:
    st.info(st.session_state.label)

# ── Dropdown for method selection ──
types = st.session_state.types_found
if types:
    methods_list = list(types.keys())
    
    # Dropdown لاختيار طريقة الحل
    selected_method = st.selectbox(
        "**Choose solution method:**",
        options=methods_list,
        index=methods_list.index(st.session_state.selected_method) if st.session_state.selected_method in methods_list else 0,
        key="method_select"
    )
    st.session_state.selected_method = selected_method
    
    # زر حل واحد
    if st.button(f"📖 Solve as {selected_method}", use_container_width=True, type="primary"):
        eq_str = st.session_state.eq_saved
        f_or_flag = types[selected_method]
        
        if selected_method == "Exact":        
            res, steps = solve_exact_with_steps(eq_str)
        elif selected_method == "Linear":     
            res, steps = solve_linear_with_steps(f_or_flag)
        elif selected_method == "Separable":  
            res, steps = solve_separable_with_steps(f_or_flag)
        elif selected_method == "Bernoulli":  
            res, steps = solve_bernoulli_with_steps(f_or_flag)
        elif selected_method == "Homogeneous":
            res, steps = solve_homogeneous_with_steps(f_or_flag)
        else: 
            res, steps = "❌ Unknown", []
        
        st.session_state.solution = res
        st.session_state.steps = steps
        st.session_state.sol_type = selected_method

# ── Solution output with steps ──
if st.session_state.solution:
    sol = st.session_state.solution
    steps = st.session_state.steps
    stype = st.session_state.sol_type
    
    if isinstance(sol, str) and sol.startswith("❌"):
        st.error(f"[{stype}] {sol}")
    else:
        st.success(f"✅ [{stype}] Solution")
        
        # عرض الخطوات
        if steps:
            with st.expander("📖 Solution Steps", expanded=True):
                for step in steps:
                    if step.startswith("**Step"):
                        st.markdown(step)
                    else:
                        st.code(step, language="python")
        
        # عرض الحل النهائي
        if sol and sol != "None":
            st.markdown("---")
            st.markdown("### 📝 Final Solution:")
            if isinstance(sol, sp.Basic):
                st.latex(sp.latex(sol))
            else:
                st.latex(sol)
