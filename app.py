import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import minimize

# -------------------- Triangular Fuzzy Number operations --------------------
def tfn_add(a, b):
    return (a[0]+b[0], a[1]+b[1], a[2]+b[2])

def tfn_mul(a, b):
    return (a[0]*b[0], a[1]*b[1], a[2]*b[2])

def tfn_div(a, b):
    return (a[0]/b[2], a[1]/b[1], a[2]/b[0])

def gmi(tfn):
    """Graded mean integration"""
    return (tfn[0] + 4*tfn[1] + tfn[2]) / 6

def geometric_mean(tfns):
    """Geometric mean of a list of TFNs"""
    prod_l = 1.0
    prod_m = 1.0
    prod_u = 1.0
    n = len(tfns)
    for t in tfns:
        prod_l *= t[0]
        prod_m *= t[1]
        prod_u *= t[2]
    return (prod_l**(1/n), prod_m**(1/n), prod_u**(1/n))

def arithmetic_mean(tfns):
    n = len(tfns)
    s_l = sum(t[0] for t in tfns)
    s_m = sum(t[1] for t in tfns)
    s_u = sum(t[2] for t in tfns)
    return (s_l/n, s_m/n, s_u/n)

# -------------------- Fuzzy Delphi --------------------
def run_delphi(criteria_tfns, threshold):
    selected = []
    agg_tfns = []
    gmi_vals = []
    for tfns in criteria_tfns:
        agg = geometric_mean(tfns)
        g = gmi(agg)
        agg_tfns.append(agg)
        gmi_vals.append(g)
        selected.append(g >= threshold)
    return selected, agg_tfns, gmi_vals

# -------------------- Fuzzy BWM --------------------
def run_bwm(factors, best_idx, worst_idx, best_to_others, others_to_worst):
    n = len(factors)
    n_exp = len(best_to_others)
    
    # Aggregate across experts using geometric mean
    agg_best = []
    agg_worst = []
    for j in range(n):
        tfns_b = [best_to_others[e][j] for e in range(n_exp)]
        agg_best.append(geometric_mean(tfns_b))
        tfns_w = [others_to_worst[e][j] for e in range(n_exp)]
        agg_worst.append(geometric_mean(tfns_w))
    
    # Initial guess
    x0 = []
    for i in range(n):
        x0.extend([1/n, 1/n, 1/n])
    x0.append(0.5)  # xi
    
    # Constraints
    cons = []
    
    # Best-to-others constraints
    for j in range(n):
        if j == best_idx:
            continue
        l_Bj, m_Bj, u_Bj = agg_best[j]
        # lower bound
        cons.append({'type': 'ineq', 'fun': lambda x, j=j, lbj=l_Bj: x[best_idx*3] - lbj * x[j*3] + x[-1]})
        cons.append({'type': 'ineq', 'fun': lambda x, j=j, lbj=l_Bj: -x[best_idx*3] + lbj * x[j*3] + x[-1]})
        # middle
        cons.append({'type': 'ineq', 'fun': lambda x, j=j, mbj=m_Bj: x[best_idx*3+1] - mbj * x[j*3+1] + x[-1]})
        cons.append({'type': 'ineq', 'fun': lambda x, j=j, mbj=m_Bj: -x[best_idx*3+1] + mbj * x[j*3+1] + x[-1]})
        # upper
        cons.append({'type': 'ineq', 'fun': lambda x, j=j, ubj=u_Bj: x[best_idx*3+2] - ubj * x[j*3+2] + x[-1]})
        cons.append({'type': 'ineq', 'fun': lambda x, j=j, ubj=u_Bj: -x[best_idx*3+2] + ubj * x[j*3+2] + x[-1]})
    
    # Others-to-worst constraints
    for j in range(n):
        if j == worst_idx:
            continue
        l_jW, m_jW, u_jW = agg_worst[j]
        cons.append({'type': 'ineq', 'fun': lambda x, j=j, ljw=l_jW: x[j*3] - ljw * x[worst_idx*3] + x[-1]})
        cons.append({'type': 'ineq', 'fun': lambda x, j=j, ljw=l_jW: -x[j*3] + ljw * x[worst_idx*3] + x[-1]})
        cons.append({'type': 'ineq', 'fun': lambda x, j=j, mjw=m_jW: x[j*3+1] - mjw * x[worst_idx*3+1] + x[-1]})
        cons.append({'type': 'ineq', 'fun': lambda x, j=j, mjw=m_jW: -x[j*3+1] + mjw * x[worst_idx*3+1] + x[-1]})
        cons.append({'type': 'ineq', 'fun': lambda x, j=j, ujw=u_jW: x[j*3+2] - ujw * x[worst_idx*3+2] + x[-1]})
        cons.append({'type': 'ineq', 'fun': lambda x, j=j, ujw=u_jW: -x[j*3+2] + ujw * x[worst_idx*3+2] + x[-1]})
    
    # Sum of GMI = 1
    def gmi_sum(x):
        s = 0
        for i in range(n):
            s += (x[i*3] + 4*x[i*3+1] + x[i*3+2]) / 6
        return s - 1
    cons.append({'type': 'eq', 'fun': gmi_sum})
    
    # Sum of m_j = 1
    def m_sum(x):
        return sum(x[i*3+1] for i in range(n)) - 1
    cons.append({'type': 'eq', 'fun': m_sum})
    
    # l_j + sum_{i≠j} u_i >= 1
    for j in range(n):
        def l_plus_u(x, j=j):
            s = x[j*3]
            for i in range(n):
                if i != j:
                    s += x[i*3+2]
            return s - 1
        cons.append({'type': 'ineq', 'fun': l_plus_u})
    
    # u_j + sum_{i≠j} l_i <= 1
    for j in range(n):
        def u_plus_l(x, j=j):
            s = x[j*3+2]
            for i in range(n):
                if i != j:
                    s += x[i*3]
            return 1 - s
        cons.append({'type': 'ineq', 'fun': u_plus_l})
    
    # l_j <= m_j <= u_j
    for j in range(n):
        cons.append({'type': 'ineq', 'fun': lambda x, j=j: x[j*3+1] - x[j*3]})
        cons.append({'type': 'ineq', 'fun': lambda x, j=j: x[j*3+2] - x[j*3+1]})
    
    # xi >= 0
    cons.append({'type': 'ineq', 'fun': lambda x: x[-1]})
    
    # Objective: minimize xi
    def objective(x):
        return x[-1]
    
    bounds = [(0, None)] * (3*n + 1)
    result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons, options={'maxiter': 1000})
    
    if not result.success:
        st.warning("Optimization failed: " + result.message)
    
    weights = []
    for i in range(n):
        l = result.x[i*3]
        m = result.x[i*3+1]
        u = result.x[i*3+2]
        weights.append((l, m, u))
    xi = result.x[-1]
    
    return weights, xi

# -------------------- Fuzzy LBWA --------------------
def run_lbwa(factors, best_idx, level_data, lambda_data):
    n = len(factors)
    n_exp = len(level_data)
    
    # Determine τ = max number of factors in any level across all experts
    level_counts = {}
    for exp_levels in level_data:
        for lvl in exp_levels:
            level_counts[lvl] = level_counts.get(lvl, 0) + 1
    tau = max(level_counts.values()) if level_counts else 1
    p = tau + 0.1  # elasticity coefficient
    
    # Compute ƛ TFN per factor (min, mean, max across experts)
    lambda_tfns = []
    for j in range(n):
        vals = [lambda_data[e][j] for e in range(n_exp)]
        l = min(vals)
        m = np.mean(vals)
        u = max(vals)
        lambda_tfns.append((l, m, u))
    
    # Use levels from first expert (consensus assumed)
    levels = level_data[0]
    
    # Influence function for each factor except best
    influence = []
    for j in range(n):
        if j == best_idx:
            influence.append((0,0,0))
            continue
        lvl = levels[j]
        lam = lambda_tfns[j]
        inf_l = p / (lvl * p + lam[2])
        inf_m = p / (lvl * p + lam[1])
        inf_u = p / (lvl * p + lam[0])
        influence.append((inf_l, inf_m, inf_u))
    
    # Weight of best factor
    sum_inf_l = 1.0
    sum_inf_m = 1.0
    sum_inf_u = 1.0
    for j in range(n):
        if j == best_idx:
            continue
        inf = influence[j]
        sum_inf_l += inf[2]
        sum_inf_m += inf[1]
        sum_inf_u += inf[0]
    
    w_best_l = 1 / sum_inf_l
    w_best_m = 1 / sum_inf_m
    w_best_u = 1 / sum_inf_u
    w_best = (w_best_l, w_best_m, w_best_u)
    
    # Weights for other factors
    weights = [None]*n
    weights[best_idx] = w_best
    for j in range(n):
        if j == best_idx:
            continue
        inf = influence[j]
        w_l = inf[0] * w_best[0]
        w_m = inf[1] * w_best[1]
        w_u = inf[2] * w_best[2]
        weights[j] = (w_l, w_m, w_u)
    
    return weights

# -------------------- Hybrid weighting --------------------
def combine_weights(fbwm_weights, lbwa_weights, alpha_tfn, beta_tfn):
    n = len(fbwm_weights)
    fbwm_l = [w[0] for w in fbwm_weights]
    fbwm_m = [w[1] for w in fbwm_weights]
    fbwm_u = [w[2] for w in fbwm_weights]
    lbwa_l = [w[0] for w in lbwa_weights]
    lbwa_m = [w[1] for w in lbwa_weights]
    lbwa_u = [w[2] for w in lbwa_weights]
    
    alpha_l, alpha_m, alpha_u = alpha_tfn
    beta_l, beta_m, beta_u = beta_tfn
    
    num_l = []
    num_m = []
    num_u = []
    for i in range(n):
        num_l.append( (fbwm_l[i] ** alpha_u) * (lbwa_l[i] ** beta_u) )
        num_m.append( (fbwm_m[i] ** alpha_m) * (lbwa_m[i] ** beta_m) )
        num_u.append( (fbwm_u[i] ** alpha_l) * (lbwa_u[i] ** beta_l) )
    
    denom_l = sum( (fbwm_l[j] ** alpha_l) * (lbwa_l[j] ** beta_l) for j in range(n) )
    denom_m = sum( (fbwm_m[j] ** alpha_m) * (lbwa_m[j] ** beta_m) for j in range(n) )
    denom_u = sum( (fbwm_u[j] ** alpha_u) * (lbwa_u[j] ** beta_u) for j in range(n) )
    
    composite = []
    for i in range(n):
        w_l = num_l[i] / denom_u
        w_m = num_m[i] / denom_m
        w_u = num_u[i] / denom_l
        composite.append((w_l, w_m, w_u))
    return composite

# -------------------- Fuzzy CoCoSo with Bonferroni --------------------
def normalize_cocoso(decision, types):
    n_alt = len(decision)
    n_crit = len(types)
    norm = [[(0,0,0) for _ in range(n_crit)] for _ in range(n_alt)]
    
    for j in range(n_crit):
        if types[j] == 'benefit':
            max_u = max(decision[i][j][2] for i in range(n_alt))
            for i in range(n_alt):
                l, m, u = decision[i][j]
                norm[i][j] = (l / max_u, m / max_u, u / max_u)
        else:  # cost
            min_l = min(decision[i][j][0] for i in range(n_alt))
            for i in range(n_alt):
                l, m, u = decision[i][j]
                norm[i][j] = (min_l / u, min_l / m, min_l / l)
    return norm

def compute_bonferroni(norm_matrix, weights, phi1, phi2):
    n_alt = len(norm_matrix)
    n_crit = len(weights)
    scob = []
    pcob = []
    
    for a in range(n_alt):
        s_l = 0
        s_m = 0
        s_u = 0
        p_l = 1.0
        p_m = 1.0
        p_u = 1.0
        
        for i in range(n_crit):
            for j in range(n_crit):
                if i == j:
                    continue
                w_i = weights[i]
                w_j = weights[j]
                gamma_i = norm_matrix[a][i]
                gamma_j = norm_matrix[a][j]
                
                # term = w_i * w_j / (1 - w_i)
                term_l = (w_i[0] * w_j[0]) / (1 - w_i[2]) if w_i[2] < 1 else 0
                term_m = (w_i[1] * w_j[1]) / (1 - w_i[1]) if w_i[1] < 1 else 0
                term_u = (w_i[2] * w_j[2]) / (1 - w_i[0]) if w_i[0] < 1 else 0
                
                s_l += term_l * (gamma_i[0]**phi1) * (gamma_j[0]**phi2)
                s_m += term_m * (gamma_i[1]**phi1) * (gamma_j[1]**phi2)
                s_u += term_u * (gamma_i[2]**phi1) * (gamma_j[2]**phi2)
                
                base_l = phi1 * gamma_i[0] + phi2 * gamma_j[0]
                base_m = phi1 * gamma_i[1] + phi2 * gamma_j[1]
                base_u = phi1 * gamma_i[2] + phi2 * gamma_j[2]
                p_l *= base_l ** term_l
                p_m *= base_m ** term_m
                p_u *= base_u ** term_u
        
        exp = 1/(phi1+phi2)
        s_l = s_l ** exp
        s_m = s_m ** exp
        s_u = s_u ** exp
        scob.append((s_l, s_m, s_u))
        
        p_l = p_l / (phi1+phi2)
        p_m = p_m / (phi1+phi2)
        p_u = p_u / (phi1+phi2)
        pcob.append((p_l, p_m, p_u))
    
    return scob, pcob

def relative_significance(scob, pcob, pi):
    n_alt = len(scob)
    sum_scob_l = sum(s[0] for s in scob)
    sum_scob_m = sum(s[1] for s in scob)
    sum_scob_u = sum(s[2] for s in scob)
    sum_pcob_l = sum(p[0] for p in pcob)
    sum_pcob_m = sum(p[1] for p in pcob)
    sum_pcob_u = sum(p[2] for p in pcob)
    
    min_scob_l = min(s[0] for s in scob)
    min_scob_m = min(s[1] for s in scob)
    min_scob_u = min(s[2] for s in scob)
    max_scob_l = max(s[0] for s in scob)
    max_scob_m = max(s[1] for s in scob)
    max_scob_u = max(s[2] for s in scob)
    
    min_pcob_l = min(p[0] for p in pcob)
    min_pcob_m = min(p[1] for p in pcob)
    min_pcob_u = min(p[2] for p in pcob)
    max_pcob_l = max(p[0] for p in pcob)
    max_pcob_m = max(p[1] for p in pcob)
    max_pcob_u = max(p[2] for p in pcob)
    
    psi_a = []
    psi_b = []
    psi_c = []
    
    for i in range(n_alt):
        s = scob[i]
        p = pcob[i]
        
        a_l = (s[0] + p[0]) / (sum_scob_u + sum_pcob_u)
        a_m = (s[1] + p[1]) / (sum_scob_m + sum_pcob_m)
        a_u = (s[2] + p[2]) / (sum_scob_l + sum_pcob_l)
        psi_a.append((a_l, a_m, a_u))
        
        b_l = s[0] / min_scob_u + p[0] / min_pcob_u
        b_m = s[1] / min_scob_m + p[1] / min_pcob_m
        b_u = s[2] / min_scob_l + p[2] / min_pcob_l
        psi_b.append((b_l, b_m, b_u))
        
        c_l = (pi * s[0] + (1-pi) * p[0]) / (pi * max_scob_u + (1-pi) * max_pcob_u)
        c_m = (pi * s[1] + (1-pi) * p[1]) / (pi * max_scob_m + (1-pi) * max_pcob_m)
        c_u = (pi * s[2] + (1-pi) * p[2]) / (pi * max_scob_l + (1-pi) * max_pcob_l)
        psi_c.append((c_l, c_m, c_u))
    
    return psi_a, psi_b, psi_c

def final_scores(psi_a, psi_b, psi_c):
    n_alt = len(psi_a)
    final = []
    for i in range(n_alt):
        a = psi_a[i]
        b = psi_b[i]
        c = psi_c[i]
        prod_l = (a[0] * b[0] * c[0]) ** (1/3)
        prod_m = (a[1] * b[1] * c[1]) ** (1/3)
        prod_u = (a[2] * b[2] * c[2]) ** (1/3)
        sum_l = (a[0] + b[0] + c[0]) / 3
        sum_m = (a[1] + b[1] + c[1]) / 3
        sum_u = (a[2] + b[2] + c[2]) / 3
        final.append((prod_l + sum_l, prod_m + sum_m, prod_u + sum_u))
    return final

# -------------------- Streamlit App --------------------
st.set_page_config(page_title="TS-to-Energy MCDM", layout="wide")
st.title("Integrated MCDM for Tannery Sludge-to-Energy Technology Selection")
st.markdown("Based on Moktadir et al. (2024), Chemical Engineering Journal")

# Sidebar navigation
module = st.sidebar.radio("Select Module", 
    ["Home", "1. Fuzzy Delphi", "2. Fuzzy BWM", "3. Fuzzy LBWA & Hybrid", "4. Fuzzy CoCoSo"])

# ---------- Home ----------
if module == "Home":
    st.header("Welcome")
    st.markdown("""
    This app implements the four‑stage MCDM framework proposed in the paper:
    - **Fuzzy Delphi** – Validate criteria using expert linguistic ratings.
    - **Fuzzy BWM** – Compute weights of main factors and sub‑factors.
    - **Fuzzy LBWA & Hybrid** – Level‑based weight assessment and combination with BWM.
    - **Fuzzy CoCoSo with Bonferroni** – Rank TS‑to‑energy technologies.
    
    Use the sidebar to navigate. You can load sample data or input your own.
    """)

# ---------- Module 1: Fuzzy Delphi ----------
elif module == "1. Fuzzy Delphi":
    st.header("Fuzzy Delphi – Criteria Validation")
    
    # Input parameters
    col1, col2 = st.columns(2)
    with col1:
        n_criteria = st.number_input("Number of criteria", min_value=1, value=5, step=1)
    with col2:
        threshold = st.number_input("Threshold (GMI)", min_value=0.0, max_value=1.0, value=0.6, step=0.05)
    
    # Linguistic scale mapping (Table S5)
    ling_map = {
        "Very Low (VLR)": (0.1, 0.1, 0.3),
        "Low (LR)": (0.1, 0.3, 0.5),
        "Medium (MR)": (0.3, 0.5, 0.7),
        "High (HR)": (0.5, 0.7, 0.9),
        "Very High (VHR)": (0.7, 0.9, 0.9)
    }
    
    # Option to load sample data
    if st.checkbox("Use sample data (5 criteria, 17 experts)"):
        # For demonstration, create synthetic sample
        # In practice, you'd load from CSV. Here we simulate.
        st.info("Loading sample data from paper (simplified).")
        # We'll just use random ratings for illustration
        experts = 17
        criteria_names = [f"C{i+1}" for i in range(n_criteria)]
        ratings = []
        for e in range(experts):
            row = []
            for c in range(n_criteria):
                # Randomly pick a linguistic term
                import random
                term = random.choice(list(ling_map.keys()))
                row.append(ling_map[term])
            ratings.append(row)
        st.write("Sample ratings generated (first 5 experts):")
        st.dataframe(pd.DataFrame(ratings[:5], columns=criteria_names))
    else:
        st.info("Enter expert ratings for each criterion.")
        experts = st.number_input("Number of experts", min_value=1, value=3, step=1)
        criteria_names = [f"C{i+1}" for i in range(n_criteria)]
        ratings = []
        for e in range(experts):
            st.subheader(f"Expert {e+1}")
            row = []
            for c in range(n_criteria):
                ling = st.selectbox(f"Criteria {criteria_names[c]}", 
                                    options=list(ling_map.keys()), key=f"e{e}_c{c}")
                row.append(ling_map[ling])
            ratings.append(row)
    
    if st.button("Run Fuzzy Delphi"):
        # Convert ratings to list of TFNs per criterion across experts
        crit_tfns = []
        for c in range(n_criteria):
            tfns = [ratings[e][c] for e in range(len(ratings))]
            crit_tfns.append(tfns)
        
        selected, agg_tfns, gmi_vals = run_delphi(crit_tfns, threshold)
        
        st.subheader("Results")
        results_df = pd.DataFrame({
            "Criterion": criteria_names,
            "Aggregated TFN": [f"({gm[0]:.3f}, {gm[1]:.3f}, {gm[2]:.3f})" for gm in agg_tfns],
            "GMI": [round(g, 4) for g in gmi_vals],
            "Selected": ["Yes" if s else "No" for s in selected]
        })
        st.dataframe(results_df)
        st.success(f"{sum(selected)} criteria selected (GMI ≥ {threshold})")

# ---------- Module 2: Fuzzy BWM ----------
elif module == "2. Fuzzy BWM":
    st.header("Fuzzy BWM – Factor Weighting")
    st.markdown("Determine the best and worst factors, then provide fuzzy comparisons.")
    
    # Linguistic scale for BWM (Table S6)
    bwm_ling_map = {
        "Very Low (VL)": (1, 1, 3),
        "Low (L)": (1, 3, 5),
        "Medium (M)": (3, 5, 7),
        "High (H)": (5, 7, 9),
        "Very High (VH)": (7, 9, 9)
    }
    
    # Input: factors
    if st.checkbox("Use sample factors (main factors from paper)"):
        factors = ["Technical (T)", "Economic (E)", "Environmental (En)", "Social (S)", "Governance (G)"]
        n_factors = len(factors)
    else:
        n_factors = st.number_input("Number of factors", min_value=2, value=3, step=1)
        factors = [st.text_input(f"Factor {i+1} name", value=f"F{i+1}") for i in range(n_factors)]
    
    n_exp = st.number_input("Number of experts", min_value=1, value=3, step=1)
    
    # Assume all experts agree on best and worst for simplicity
    best_idx = st.selectbox("Best factor", options=range(n_factors), format_func=lambda x: factors[x])
    worst_idx = st.selectbox("Worst factor", options=range(n_factors), format_func=lambda x: factors[x])
    
    # Collect best-to-others and others-to-worst for each expert
    if st.button("Proceed to input comparisons"):
        st.session_state['bwm_factors'] = factors
        st.session_state['bwm_best'] = best_idx
        st.session_state['bwm_worst'] = worst_idx
        st.session_state['bwm_n_exp'] = n_exp
    
    if 'bwm_factors' in st.session_state:
        factors = st.session_state['bwm_factors']
        best_idx = st.session_state['bwm_best']
        worst_idx = st.session_state['bwm_worst']
        n_exp = st.session_state['bwm_n_exp']
        
        st.subheader("Best-to-Others Comparisons")
        best_to_others = []
        for e in range(n_exp):
            st.write(f"**Expert {e+1}**")
            row = []
            for j in range(len(factors)):
                if j == best_idx:
                    row.append((1,1,1))
                else:
                    ling = st.selectbox(f"Best to {factors[j]}", 
                                        options=list(bwm_ling_map.keys()),
                                        key=f"bto_e{e}_f{j}")
                    row.append(bwm_ling_map[ling])
            best_to_others.append(row)
        
        st.subheader("Others-to-Worst Comparisons")
        others_to_worst = []
        for e in range(n_exp):
            st.write(f"**Expert {e+1}**")
            row = []
            for j in range(len(factors)):
                if j == worst_idx:
                    row.append((1,1,1))
                else:
                    ling = st.selectbox(f"{factors[j]} to worst", 
                                        options=list(bwm_ling_map.keys()),
                                        key=f"otw_e{e}_f{j}")
                    row.append(bwm_ling_map[ling])
            others_to_worst.append(row)
        
        if st.button("Compute Fuzzy BWM Weights"):
            weights, xi = run_bwm(factors, best_idx, worst_idx, best_to_others, others_to_worst)
            st.subheader("Fuzzy Weights")
            for i, w in enumerate(weights):
                st.write(f"{factors[i]}: ({w[0]:.4f}, {w[1]:.4f}, {w[2]:.4f})  GMI = {gmi(w):.4f}")
            st.info(f"Consistency index ξ = {xi:.4f}")
            st.session_state['bwm_weights'] = weights

# ---------- Module 3: Fuzzy LBWA & Hybrid ----------
elif module == "3. Fuzzy LBWA & Hybrid":
    st.header("Fuzzy LBWA and Hybrid Weighting")
    
    if st.checkbox("Use same factors as in BWM"):
        if 'bwm_factors' in st.session_state:
            factors = st.session_state['bwm_factors']
            best_idx = st.session_state['bwm_best']
        else:
            st.warning("Please run BWM first or input manually.")
            factors = ["T", "E", "En", "S", "G"]
            best_idx = 1
    else:
        n_factors = st.number_input("Number of factors", min_value=2, value=5, step=1)
        factors = [st.text_input(f"Factor {i+1} name", value=f"F{i+1}") for i in range(n_factors)]
        best_idx = st.selectbox("Best factor", options=range(n_factors), format_func=lambda x: factors[x])
    
    n_exp = st.number_input("Number of experts", min_value=1, value=5, step=1)
    
    st.markdown("For each factor, assign a **level** (1,2,...) and a **comparison value ƛ** (0–τ).")
    st.markdown("τ = maximum number of factors in any level (auto‑computed).")
    
    # Create input tables
    level_data = []
    lambda_data = []
    for e in range(n_exp):
        st.subheader(f"Expert {e+1}")
        levels = []
        lambdas = []
        for j, f in enumerate(factors):
            col1, col2 = st.columns(2)
            with col1:
                lvl = st.number_input(f"Level of {f}", min_value=1, value=1, step=1, key=f"lvl_e{e}_f{j}")
            with col2:
                lam = st.number_input(f"ƛ for {f}", min_value=0.0, max_value=10.0, value=0.0, step=0.1, key=f"lam_e{e}_f{j}")
            levels.append(lvl)
            lambdas.append(lam)
        level_data.append(levels)
        lambda_data.append(lambdas)
    
    # Priority factors for hybrid (α, β) – linguistic scale Table S7
    hybrid_scale = {
        "Very Low (VL)": (0, 0, 0.16),
        "Low (L)": (0, 0.16, 0.34),
        "Medium Low (ML)": (0.16, 0.34, 0.5),
        "Moderate (M)": (0.34, 0.5, 0.66),
        "Medium High (MH)": (0.5, 0.66, 0.84),
        "High (H)": (0.66, 0.84, 1),
        "Very High (VH)": (0.84, 1, 1)
    }
    st.subheader("Hybrid Priority Factors")
    alpha_ling = st.selectbox("Priority of FBWM weight (α)", options=list(hybrid_scale.keys()), index=3)
    beta_ling = st.selectbox("Priority of FLBWA weight (β)", options=list(hybrid_scale.keys()), index=3)
    alpha_tfn = hybrid_scale[alpha_ling]
    beta_tfn = hybrid_scale[beta_ling]
    
    if st.button("Compute LBWA and Hybrid Weights"):
        lbwa_weights = run_lbwa(factors, best_idx, level_data, lambda_data)
        st.subheader("Fuzzy LBWA Weights")
        for i, w in enumerate(lbwa_weights):
            st.write(f"{factors[i]}: ({w[0]:.4f}, {w[1]:.4f}, {w[2]:.4f})  GMI = {gmi(w):.4f}")
        
        if 'bwm_weights' in st.session_state:
            bwm_weights = st.session_state['bwm_weights']
            hybrid_weights = combine_weights(bwm_weights, lbwa_weights, alpha_tfn, beta_tfn)
            st.subheader("Hybrid (FBWM + FLBWA) Weights")
            for i, w in enumerate(hybrid_weights):
                st.write(f"{factors[i]}: ({w[0]:.4f}, {w[1]:.4f}, {w[2]:.4f})  GMI = {gmi(w):.4f}")
            st.session_state['hybrid_weights'] = hybrid_weights
        else:
            st.warning("BWM weights not found. Using LBWA only.")
            st.session_state['hybrid_weights'] = lbwa_weights

# ---------- Module 4: Fuzzy CoCoSo with Bonferroni ----------
elif module == "4. Fuzzy CoCoSo":
    st.header("Fuzzy CoCoSo with Bonferroni – Technology Ranking")
    
    # Input alternatives
    alt_names = ["TS-SS", "TS-HP", "TS-PCC", "TS-MPP"] if st.checkbox("Use sample alternatives") else \
                [x.strip() for x in st.text_area("Alternative names (one per line)").splitlines() if x.strip()]
    
    # Criteria and weights from previous modules or input
    if 'hybrid_weights' in st.session_state:
        weights = st.session_state['hybrid_weights']
        criteria_names = st.session_state.get('bwm_factors', [f"C{i+1}" for i in range(len(weights))])
    else:
        st.warning("No weights found. Please input manually.")
        n_crit = st.number_input("Number of criteria", min_value=1, value=5)
        criteria_names = [f"C{i+1}" for i in range(n_crit)]
        weights = []
        for i in range(n_crit):
            col1, col2, col3 = st.columns(3)
            with col1:
                l = st.number_input(f"Weight {criteria_names[i]} (l)", value=0.1, key=f"w{i}_l")
            with col2:
                m = st.number_input(f"Weight {criteria_names[i]} (m)", value=0.2, key=f"w{i}_m")
            with col3:
                u = st.number_input(f"Weight {criteria_names[i]} (u)", value=0.3, key=f"w{i}_u")
            weights.append((l, m, u))
    
    # Criteria types
    types = []
    for i, c in enumerate(criteria_names):
        typ = st.selectbox(f"Type of {c}", ["benefit", "cost"], key=f"type_{i}")
        types.append(typ)
    
    # Decision matrix
    if st.checkbox("Use sample decision matrix (from paper)"):
        # For demonstration, create synthetic data
        st.info("Loading sample aggregated matrix (Table S45).")
        # We'll generate random TFNs for illustration
        n_alt = len(alt_names)
        n_crit = len(criteria_names)
        decision = []
        for a in range(n_alt):
            alt_row = []
            for j in range(n_crit):
                # Random TFN between 0 and 10
                l = np.random.uniform(0, 5)
                m = np.random.uniform(l, 8)
                u = np.random.uniform(m, 10)
                alt_row.append((l, m, u))
            decision.append(alt_row)
    else:
        st.info("Enter decision matrix as TFNs for each alternative and criterion.")
        decision = []
        for a, alt in enumerate(alt_names):
            st.subheader(alt)
            alt_row = []
            for j, c in enumerate(criteria_names):
                col1, col2, col3 = st.columns(3)
                with col1:
                    l = st.number_input(f"{c} l", value=0.0, key=f"d{a}_{j}_l")
                with col2:
                    m = st.number_input(f"{c} m", value=0.5, key=f"d{a}_{j}_m")
                with col3:
                    u = st.number_input(f"{c} u", value=1.0, key=f"d{a}_{j}_u")
                alt_row.append((l, m, u))
            decision.append(alt_row)
    
    # Parameters
    col1, col2, col3 = st.columns(3)
    with col1:
        phi1 = st.number_input("ϕ1", value=1.0, step=0.1)
        phi2 = st.number_input("ϕ2", value=1.0, step=0.1)
    with col2:
        pi = st.number_input("π (coefficient)", value=0.5, min_value=0.0, max_value=1.0, step=0.05)
    with col3:
        sigma = st.number_input("σ (uncertainty for quant)", value=0.1, min_value=0.0, max_value=0.5, step=0.05)
        st.caption("Not used in this simplified version")
    
    if st.button("Run Fuzzy CoCoSo"):
        norm_matrix = normalize_cocoso(decision, types)
        scob, pcob = compute_bonferroni(norm_matrix, weights, phi1, phi2)
        psi_a, psi_b, psi_c = relative_significance(scob, pcob, pi)
        final = final_scores(psi_a, psi_b, psi_c)
        
        rankings = sorted([(i, gmi(final[i])) for i in range(len(alt_names))], 
                          key=lambda x: x[1], reverse=True)
        
        st.subheader("Ranking Results")
        for rank, (idx, score) in enumerate(rankings, start=1):
            st.write(f"{rank}. {alt_names[idx]} – GMI = {score:.4f}")
