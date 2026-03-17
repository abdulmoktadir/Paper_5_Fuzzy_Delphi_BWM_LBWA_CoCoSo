import math
import numpy as np
import pandas as pd
import streamlit as st
from scipy.optimize import minimize

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Integrated Fuzzy MCDM Platform",
    page_icon="📊",
    layout="wide",
)

# ============================================================
# STYLES
# ============================================================
CSS = """
<style>
.block-container {
    padding-top: 1.1rem;
    padding-bottom: 2rem;
    max-width: 1450px;
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #f7faff 0%, #eef4ff 100%);
}
.app-card {
    background: #ffffff;
    border: 1px solid #e8eef7;
    border-radius: 16px;
    padding: 1rem 1.1rem;
    box-shadow: 0 2px 10px rgba(0,0,0,0.04);
    margin-bottom: 0.8rem;
}
.card-title {
    font-weight: 700;
    font-size: 1.05rem;
    margin-bottom: 0.2rem;
}
.small-note {
    color: #5f6b7a;
    font-size: 0.92rem;
}
.section-head {
    padding: 0.45rem 0.8rem;
    border-left: 5px solid #4f7cff;
    background: #f5f8ff;
    border-radius: 10px;
    margin: 0.25rem 0 1rem 0;
    font-weight: 700;
}
.metric-box {
    background: #f8fbff;
    border: 1px solid #e5edf7;
    border-radius: 14px;
    padding: 0.75rem 1rem;
}
code {
    color: #0b5ed7;
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

EPS = 1e-12

# ============================================================
# SCALES
# ============================================================
DELHI_SCALE = {
    "VLR": (0.1, 0.1, 0.3),
    "LR":  (0.1, 0.3, 0.5),
    "MR":  (0.3, 0.5, 0.7),
    "HR":  (0.5, 0.7, 0.9),
    "VHR": (0.7, 0.9, 0.9),
}

BWM_SCALE = {
    "EQ": (1, 1, 1),
    "VL": (1, 1, 3),
    "L":  (1, 3, 5),
    "M":  (3, 5, 7),
    "H":  (5, 7, 9),
    "VH": (7, 9, 9),
}

HYBRID_SCALE = {
    "VL": (0.00, 0.00, 0.16),
    "L":  (0.00, 0.16, 0.34),
    "ML": (0.16, 0.34, 0.50),
    "M":  (0.34, 0.50, 0.66),
    "MH": (0.50, 0.66, 0.84),
    "H":  (0.66, 0.84, 1.00),
    "VH": (0.84, 1.00, 1.00),
}

DELHI_MEANING = {
    "VLR": "Very Low Relevance",
    "LR": "Low Relevance",
    "MR": "Moderate Relevance",
    "HR": "High Relevance",
    "VHR": "Very High Relevance",
}

BWM_MEANING = {
    "EQ": "Equal",
    "VL": "Very Low",
    "L": "Low",
    "M": "Medium",
    "H": "High",
    "VH": "Very High",
}

HYBRID_MEANING = {
    "VL": "Very Low",
    "L": "Low",
    "ML": "Medium Low",
    "M": "Moderate",
    "MH": "Medium High",
    "H": "High",
    "VH": "Very High",
}

# ============================================================
# UTILITY FUNCTIONS
# ============================================================
def gmi(tfn):
    return (tfn[0] + 4 * tfn[1] + tfn[2]) / 6

def geometric_mean(tfns):
    prod_l, prod_m, prod_u = 1.0, 1.0, 1.0
    n = len(tfns)
    for t in tfns:
        prod_l *= max(t[0], EPS)
        prod_m *= max(t[1], EPS)
        prod_u *= max(t[2], EPS)
    return (prod_l ** (1 / n), prod_m ** (1 / n), prod_u ** (1 / n))

def arithmetic_mean(tfns):
    n = len(tfns)
    return (
        sum(t[0] for t in tfns) / n,
        sum(t[1] for t in tfns) / n,
        sum(t[2] for t in tfns) / n,
    )

def tfn_to_str(t):
    return f"({t[0]:.4f}, {t[1]:.4f}, {t[2]:.4f})"

def parse_names(text, n, prefix):
    names = [x.strip() for x in text.splitlines() if x.strip()]
    if len(names) == 0:
        names = [f"{prefix}{i+1}" for i in range(n)]
    elif len(names) < n:
        names += [f"{prefix}{i+1}" for i in range(len(names), n)]
    return names[:n]

def render_scale_table(scale_dict, meaning_dict, title):
    df = pd.DataFrame({
        "Code": list(scale_dict.keys()),
        "Meaning": [meaning_dict[k] for k in scale_dict.keys()],
        "TFN": [str(scale_dict[k]) for k in scale_dict.keys()]
    })
    with st.expander(title, expanded=False):
        st.dataframe(df, use_container_width=True, hide_index=True)

def make_bar_df(items, scores, item_col="Item", score_col="GMI"):
    return pd.DataFrame({item_col: items, score_col: scores}).set_index(item_col)

# ============================================================
# FUZZY DELPHI
# ============================================================
def run_delphi(criteria_tfns, threshold):
    selected, agg_tfns, gmi_vals = [], [], []
    for tfns in criteria_tfns:
        agg = geometric_mean(tfns)
        val = gmi(agg)
        agg_tfns.append(agg)
        gmi_vals.append(val)
        selected.append(val >= threshold)
    return selected, agg_tfns, gmi_vals

# ============================================================
# FUZZY BWM
# ============================================================
def solve_bwm_aggregated(agg_best, agg_worst, best_idx, worst_idx):
    n = len(agg_best)
    x0 = []
    for _ in range(n):
        x0.extend([1 / n, 1 / n, 1 / n])
    x0.append(0.5)  # xi

    cons = []

    # Best-to-others constraints
    for j in range(n):
        if j == best_idx:
            continue
        l_Bj, m_Bj, u_Bj = agg_best[j]
        cons.append({'type': 'ineq', 'fun': lambda x, j=j, lbj=l_Bj: x[best_idx*3]   - lbj * x[j*3]   + x[-1]})
        cons.append({'type': 'ineq', 'fun': lambda x, j=j, lbj=l_Bj: -x[best_idx*3]  + lbj * x[j*3]   + x[-1]})
        cons.append({'type': 'ineq', 'fun': lambda x, j=j, mbj=m_Bj: x[best_idx*3+1] - mbj * x[j*3+1] + x[-1]})
        cons.append({'type': 'ineq', 'fun': lambda x, j=j, mbj=m_Bj: -x[best_idx*3+1]+ mbj * x[j*3+1] + x[-1]})
        cons.append({'type': 'ineq', 'fun': lambda x, j=j, ubj=u_Bj: x[best_idx*3+2] - ubj * x[j*3+2] + x[-1]})
        cons.append({'type': 'ineq', 'fun': lambda x, j=j, ubj=u_Bj: -x[best_idx*3+2]+ ubj * x[j*3+2] + x[-1]})

    # Others-to-worst constraints
    for j in range(n):
        if j == worst_idx:
            continue
        l_jW, m_jW, u_jW = agg_worst[j]
        cons.append({'type': 'ineq', 'fun': lambda x, j=j, ljw=l_jW: x[j*3]   - ljw * x[worst_idx*3]   + x[-1]})
        cons.append({'type': 'ineq', 'fun': lambda x, j=j, ljw=l_jW: -x[j*3]  + ljw * x[worst_idx*3]   + x[-1]})
        cons.append({'type': 'ineq', 'fun': lambda x, j=j, mjw=m_jW: x[j*3+1] - mjw * x[worst_idx*3+1] + x[-1]})
        cons.append({'type': 'ineq', 'fun': lambda x, j=j, mjw=m_jW: -x[j*3+1]+ mjw * x[worst_idx*3+1] + x[-1]})
        cons.append({'type': 'ineq', 'fun': lambda x, j=j, ujw=u_jW: x[j*3+2] - ujw * x[worst_idx*3+2] + x[-1]})
        cons.append({'type': 'ineq', 'fun': lambda x, j=j, ujw=u_jW: -x[j*3+2]+ ujw * x[worst_idx*3+2] + x[-1]})

    # Sum of GMI = 1
    def gmi_sum(x):
        return sum((x[i*3] + 4*x[i*3+1] + x[i*3+2]) / 6 for i in range(n)) - 1
    cons.append({'type': 'eq', 'fun': gmi_sum})

    # Sum of middle values = 1
    def m_sum(x):
        return sum(x[i*3+1] for i in range(n)) - 1
    cons.append({'type': 'eq', 'fun': m_sum})

    # l_j + sum_{i!=j} u_i >= 1
    for j in range(n):
        def l_plus_u(x, j=j):
            s = x[j*3]
            for i in range(n):
                if i != j:
                    s += x[i*3+2]
            return s - 1
        cons.append({'type': 'ineq', 'fun': l_plus_u})

    # u_j + sum_{i!=j} l_i <= 1
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

    def objective(x):
        return x[-1]

    bounds = [(0, None)] * (3 * n + 1)
    result = minimize(
        objective,
        x0,
        method='SLSQP',
        bounds=bounds,
        constraints=cons,
        options={'maxiter': 1000}
    )

    if not result.success:
        st.warning("Optimization warning: " + result.message)

    weights = []
    for i in range(n):
        l = result.x[i*3]
        m = result.x[i*3+1]
        u = result.x[i*3+2]
        weights.append((l, m, u))
    xi = result.x[-1]
    return weights, xi

# ============================================================
# FUZZY LBWA
# ============================================================
def run_lbwa(factors, best_idx, level_data, lambda_data):
    n = len(factors)
    n_exp = len(level_data)

    level_counts = {}
    for exp_levels in level_data:
        for lvl in exp_levels:
            level_counts[lvl] = level_counts.get(lvl, 0) + 1
    tau = max(level_counts.values()) if level_counts else 1
    p = tau + 0.1

    lambda_tfns = []
    for j in range(n):
        vals = [lambda_data[e][j] for e in range(n_exp)]
        l = min(vals)
        m = float(np.mean(vals))
        u = max(vals)
        lambda_tfns.append((l, m, u))

    levels = level_data[0]

    influence = []
    for j in range(n):
        if j == best_idx:
            influence.append((0, 0, 0))
            continue
        lvl = max(levels[j], 1)
        lam = lambda_tfns[j]
        inf_l = p / (lvl * p + lam[2] + EPS)
        inf_m = p / (lvl * p + lam[1] + EPS)
        inf_u = p / (lvl * p + lam[0] + EPS)
        influence.append((inf_l, inf_m, inf_u))

    sum_inf_l, sum_inf_m, sum_inf_u = 1.0, 1.0, 1.0
    for j in range(n):
        if j == best_idx:
            continue
        inf = influence[j]
        sum_inf_l += inf[2]
        sum_inf_m += inf[1]
        sum_inf_u += inf[0]

    w_best = (1 / sum_inf_l, 1 / sum_inf_m, 1 / sum_inf_u)

    weights = [None] * n
    weights[best_idx] = w_best
    for j in range(n):
        if j == best_idx:
            continue
        inf = influence[j]
        weights[j] = (
            inf[0] * w_best[0],
            inf[1] * w_best[1],
            inf[2] * w_best[2],
        )
    return weights

# ============================================================
# HYBRID COMBINATION
# ============================================================
def combine_weights(fbwm_weights, lbwa_weights, alpha_tfn, beta_tfn):
    n = len(fbwm_weights)

    alpha_l, alpha_m, alpha_u = alpha_tfn
    beta_l, beta_m, beta_u = beta_tfn

    num_l, num_m, num_u = [], [], []
    for i in range(n):
        num_l.append((max(fbwm_weights[i][0], EPS) ** alpha_u) * (max(lbwa_weights[i][0], EPS) ** beta_u))
        num_m.append((max(fbwm_weights[i][1], EPS) ** alpha_m) * (max(lbwa_weights[i][1], EPS) ** beta_m))
        num_u.append((max(fbwm_weights[i][2], EPS) ** alpha_l) * (max(lbwa_weights[i][2], EPS) ** beta_l))

    denom_l = sum((max(fbwm_weights[j][0], EPS) ** alpha_l) * (max(lbwa_weights[j][0], EPS) ** beta_l) for j in range(n))
    denom_m = sum((max(fbwm_weights[j][1], EPS) ** alpha_m) * (max(lbwa_weights[j][1], EPS) ** beta_m) for j in range(n))
    denom_u = sum((max(fbwm_weights[j][2], EPS) ** alpha_u) * (max(lbwa_weights[j][2], EPS) ** beta_u) for j in range(n))

    composite = []
    for i in range(n):
        w_l = num_l[i] / max(denom_u, EPS)
        w_m = num_m[i] / max(denom_m, EPS)
        w_u = num_u[i] / max(denom_l, EPS)
        composite.append((w_l, w_m, w_u))
    return composite

# ============================================================
# FUZZY COCO-SO WITH BONFERRONI
# ============================================================
def normalize_cocoso(decision, types):
    n_alt = len(decision)
    n_crit = len(types)
    norm = [[(0, 0, 0) for _ in range(n_crit)] for _ in range(n_alt)]

    for j in range(n_crit):
        if types[j] == "benefit":
            max_u = max(decision[i][j][2] for i in range(n_alt))
            max_u = max(max_u, EPS)
            for i in range(n_alt):
                l, m, u = decision[i][j]
                norm[i][j] = (l / max_u, m / max_u, u / max_u)
        else:
            min_l = min(decision[i][j][0] for i in range(n_alt))
            min_l = max(min_l, EPS)
            for i in range(n_alt):
                l, m, u = decision[i][j]
                norm[i][j] = (
                    min_l / max(u, EPS),
                    min_l / max(m, EPS),
                    min_l / max(l, EPS),
                )
    return norm

def compute_bonferroni(norm_matrix, weights, phi1, phi2):
    n_alt = len(norm_matrix)
    n_crit = len(weights)
    scob, pcob = [], []

    exp_val = 1 / max(phi1 + phi2, EPS)

    for a in range(n_alt):
        s_l = s_m = s_u = 0.0
        p_l = p_m = p_u = 1.0

        for i in range(n_crit):
            for j in range(n_crit):
                if i == j:
                    continue

                w_i = weights[i]
                w_j = weights[j]
                gamma_i = norm_matrix[a][i]
                gamma_j = norm_matrix[a][j]

                term_l = (w_i[0] * w_j[0]) / max(1 - w_i[2], EPS)
                term_m = (w_i[1] * w_j[1]) / max(1 - w_i[1], EPS)
                term_u = (w_i[2] * w_j[2]) / max(1 - w_i[0], EPS)

                s_l += term_l * (max(gamma_i[0], EPS) ** phi1) * (max(gamma_j[0], EPS) ** phi2)
                s_m += term_m * (max(gamma_i[1], EPS) ** phi1) * (max(gamma_j[1], EPS) ** phi2)
                s_u += term_u * (max(gamma_i[2], EPS) ** phi1) * (max(gamma_j[2], EPS) ** phi2)

                base_l = max(phi1 * gamma_i[0] + phi2 * gamma_j[0], EPS)
                base_m = max(phi1 * gamma_i[1] + phi2 * gamma_j[1], EPS)
                base_u = max(phi1 * gamma_i[2] + phi2 * gamma_j[2], EPS)

                p_l *= base_l ** term_l
                p_m *= base_m ** term_m
                p_u *= base_u ** term_u

        scob.append((s_l ** exp_val, s_m ** exp_val, s_u ** exp_val))
        pcob.append((
            p_l / max(phi1 + phi2, EPS),
            p_m / max(phi1 + phi2, EPS),
            p_u / max(phi1 + phi2, EPS),
        ))

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

    psi_a, psi_b, psi_c = [], [], []

    for i in range(n_alt):
        s = scob[i]
        p = pcob[i]

        a_l = (s[0] + p[0]) / max(sum_scob_u + sum_pcob_u, EPS)
        a_m = (s[1] + p[1]) / max(sum_scob_m + sum_pcob_m, EPS)
        a_u = (s[2] + p[2]) / max(sum_scob_l + sum_pcob_l, EPS)
        psi_a.append((a_l, a_m, a_u))

        b_l = s[0] / max(min_scob_u, EPS) + p[0] / max(min_pcob_u, EPS)
        b_m = s[1] / max(min_scob_m, EPS) + p[1] / max(min_pcob_m, EPS)
        b_u = s[2] / max(min_scob_l, EPS) + p[2] / max(min_pcob_l, EPS)
        psi_b.append((b_l, b_m, b_u))

        c_l = (pi * s[0] + (1 - pi) * p[0]) / max(pi * max_scob_u + (1 - pi) * max_pcob_u, EPS)
        c_m = (pi * s[1] + (1 - pi) * p[1]) / max(pi * max_scob_m + (1 - pi) * max_pcob_m, EPS)
        c_u = (pi * s[2] + (1 - pi) * p[2]) / max(pi * max_scob_l + (1 - pi) * max_pcob_l, EPS)
        psi_c.append((c_l, c_m, c_u))

    return psi_a, psi_b, psi_c

def final_scores(psi_a, psi_b, psi_c):
    n_alt = len(psi_a)
    final = []
    for i in range(n_alt):
        a, b, c = psi_a[i], psi_b[i], psi_c[i]
        prod_l = (max(a[0], EPS) * max(b[0], EPS) * max(c[0], EPS)) ** (1 / 3)
        prod_m = (max(a[1], EPS) * max(b[1], EPS) * max(c[1], EPS)) ** (1 / 3)
        prod_u = (max(a[2], EPS) * max(b[2], EPS) * max(c[2], EPS)) ** (1 / 3)
        sum_l = (a[0] + b[0] + c[0]) / 3
        sum_m = (a[1] + b[1] + c[1]) / 3
        sum_u = (a[2] + b[2] + c[2]) / 3
        final.append((prod_l + sum_l, prod_m + sum_m, prod_u + sum_u))
    return final

# ============================================================
# SAMPLE INITIALIZERS
# ============================================================
def init_delphi_table(criteria_names, n_exp, use_sample=False):
    rng = np.random.default_rng(42)
    expert_cols = [f"E{i+1}" for i in range(n_exp)]
    if use_sample:
        vals = rng.choice(list(DELHI_SCALE.keys()), size=(len(criteria_names), n_exp), replace=True)
        return pd.DataFrame(vals, index=criteria_names, columns=expert_cols)
    return pd.DataFrame("MR", index=criteria_names, columns=expert_cols)

def init_bwm_summary(factors, n_exp, use_sample=False):
    rng = np.random.default_rng(123)
    rows = []
    for _ in range(n_exp):
        if use_sample:
            best_idx = int(rng.integers(0, len(factors)))
            worst_idx = int(rng.integers(0, len(factors)-1))
            if worst_idx >= best_idx:
                worst_idx += 1
            rows.append([factors[best_idx], factors[worst_idx]])
        else:
            rows.append([factors[0], factors[-1]])
    return pd.DataFrame(rows, columns=["Best", "Worst"], index=[f"E{i+1}" for i in range(n_exp)])

def init_bwm_pairwise_df(factors, best_factor, worst_factor, use_sample=False, seed=0):
    rng = np.random.default_rng(seed)
    codes = ["VL", "L", "M", "H", "VH"]
    if use_sample:
        b_codes = rng.choice(codes, size=len(factors))
        w_codes = rng.choice(codes, size=len(factors))
    else:
        b_codes = np.array(["M"] * len(factors))
        w_codes = np.array(["M"] * len(factors))

    df = pd.DataFrame({
        "Factor": factors,
        "B→j": b_codes,
        "j→W": w_codes,
    })
    df.loc[df["Factor"] == best_factor, "B→j"] = "EQ"
    df.loc[df["Factor"] == worst_factor, "j→W"] = "EQ"
    return df

def init_lbwa_df(factors, use_sample=False, seed=0):
    rng = np.random.default_rng(seed)
    if use_sample:
        levels = rng.integers(1, 4, size=len(factors))
        lambdas = np.round(rng.uniform(0.0, 2.0, size=len(factors)), 2)
    else:
        levels = np.ones(len(factors), dtype=int)
        lambdas = np.zeros(len(factors))
    return pd.DataFrame({
        "Factor": factors,
        "Level": levels,
        "λ": lambdas
    })

def init_criteria_weight_df(criteria_names, weights=None):
    if weights is None:
        weights = [(0.1, 0.2, 0.3)] * len(criteria_names)
    return pd.DataFrame({
        "Criterion": criteria_names,
        "w_l": [w[0] for w in weights],
        "w_m": [w[1] for w in weights],
        "w_u": [w[2] for w in weights],
        "Type": ["benefit"] * len(criteria_names),
    })

def init_decision_df(criteria_names, use_sample=False, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for c in criteria_names:
        if use_sample:
            l = round(float(rng.uniform(0.5, 5.0)), 3)
            m = round(float(rng.uniform(l, 7.5)), 3)
            u = round(float(rng.uniform(m, 9.5)), 3)
        else:
            l, m, u = 0.0, 0.5, 1.0
        rows.append([c, l, m, u])
    return pd.DataFrame(rows, columns=["Criterion", "l", "m", "u"])

# ============================================================
# SIDEBAR NAVIGATOR
# ============================================================
st.sidebar.title("📘 Model Navigator")
module = st.sidebar.radio(
    "Choose model",
    [
        "🏠 Home",
        "1) Fuzzy Delphi",
        "2) Fuzzy BWM",
        "3) Fuzzy LBWA + Hybrid",
        "4) Fuzzy CoCoSo",
    ],
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    **Workflow**
    
    1. Fuzzy Delphi  
    2. Fuzzy BWM  
    3. Fuzzy LBWA + Hybrid  
    4. Fuzzy CoCoSo  
    """
)
st.sidebar.caption("Use abbreviations in the input tables for faster entry.")

# ============================================================
# HEADER
# ============================================================
st.title("Integrated Fuzzy MCDM Platform")
st.caption("Cleaner interface for Fuzzy Delphi, Fuzzy BWM, Fuzzy LBWA-Hybrid, and Fuzzy CoCoSo")

# ============================================================
# HOME
# ============================================================
if module == "🏠 Home":
    st.markdown('<div class="section-head">Overview</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            """
            <div class="app-card">
                <div class="card-title">What this app does</div>
                <div class="small-note">
                This interface supports a full multi-stage fuzzy MCDM workflow:
                criteria screening, factor weighting, hybrid weighting, and final technology ranking.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            """
            <div class="app-card">
                <div class="card-title">Input style</div>
                <div class="small-note">
                Major data-entry sections now use compact table editors.  
                Fuzzy Delphi uses only code inputs such as <code>VLR</code>, <code>LR</code>, <code>MR</code>, <code>HR</code>, <code>VHR</code>.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    a1, a2, a3, a4 = st.columns(4)
    with a1:
        st.metric("Stage 1", "Fuzzy Delphi")
    with a2:
        st.metric("Stage 2", "Fuzzy BWM")
    with a3:
        st.metric("Stage 3", "LBWA + Hybrid")
    with a4:
        st.metric("Stage 4", "Fuzzy CoCoSo")

    st.markdown(
        """
        <div class="app-card">
            <div class="card-title">Recommended sequence</div>
            <div class="small-note">
            Start from Fuzzy Delphi, then compute BWM weights, combine with LBWA if needed,
            and finally run CoCoSo for ranking.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ============================================================
# MODULE 1: FUZZY DELPHI
# ============================================================
elif module == "1) Fuzzy Delphi":
    st.markdown('<div class="section-head">Fuzzy Delphi – Criteria Validation</div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns([1, 1, 1, 1.2])
    with c1:
        n_criteria = st.number_input("Number of criteria", min_value=1, value=5, step=1)
    with c2:
        n_exp = st.number_input("Number of experts", min_value=1, value=5, step=1)
    with c3:
        threshold = st.number_input("Threshold (GMI)", min_value=0.0, max_value=1.0, value=0.60, step=0.05)
    with c4:
        use_sample_delphi = st.toggle("Use sample table", value=False)

    criteria_text = st.text_area(
        "Criterion names (one per line)",
        value="\n".join([f"C{i+1}" for i in range(n_criteria)]),
        height=120,
    )
    criteria_names = parse_names(criteria_text, n_criteria, "C")

    render_scale_table(DELHI_SCALE, DELHI_MEANING, "Show Delphi code legend")

    delphi_sig = ("|".join(criteria_names), n_exp, use_sample_delphi)
    if st.button("Prepare / Refresh Delphi Table", key="prep_delphi") or st.session_state.get("delphi_sig") != delphi_sig:
        st.session_state["delphi_sig"] = delphi_sig
        st.session_state["delphi_df"] = init_delphi_table(criteria_names, n_exp, use_sample_delphi)

    delphi_df = st.session_state["delphi_df"]

    edited_delphi = st.data_editor(
        delphi_df,
        key="delphi_editor",
        use_container_width=True,
        num_rows="fixed",
        column_config={
            col: st.column_config.SelectboxColumn(
                col,
                options=list(DELHI_SCALE.keys()),
                required=True,
                help="Use only abbreviation codes"
            )
            for col in delphi_df.columns
        }
    )

    if st.button("Run Fuzzy Delphi", type="primary"):
        criteria_tfns = []
        for crit in edited_delphi.index:
            tfns = [DELHI_SCALE[str(edited_delphi.loc[crit, col])] for col in edited_delphi.columns]
            criteria_tfns.append(tfns)

        selected, agg_tfns, gmi_vals = run_delphi(criteria_tfns, threshold)

        results_df = pd.DataFrame({
            "Criterion": criteria_names,
            "Aggregated TFN": [tfn_to_str(x) for x in agg_tfns],
            "GMI": [round(x, 4) for x in gmi_vals],
            "Selected": ["Yes" if x else "No" for x in selected],
        })

        s1, s2, s3 = st.columns(3)
        with s1:
            st.metric("Total criteria", len(criteria_names))
        with s2:
            st.metric("Selected", int(sum(selected)))
        with s3:
            st.metric("Rejected", int(len(criteria_names) - sum(selected)))

        st.dataframe(results_df, use_container_width=True, hide_index=True)
        st.bar_chart(make_bar_df(criteria_names, gmi_vals, "Criterion", "GMI"))
        st.success(f"{sum(selected)} criteria satisfied the threshold ≥ {threshold:.2f}")

# ============================================================
# MODULE 2: FUZZY BWM
# ============================================================
elif module == "2) Fuzzy BWM":
    st.markdown('<div class="section-head">Fuzzy BWM – Factor Weighting</div>', unsafe_allow_html=True)

    c1, c2 = st.columns([1, 1.4])
    with c1:
        use_sample_bwm = st.toggle("Use sample factors", value=True)
        n_exp_bwm = st.number_input("Number of experts", min_value=1, value=4, step=1)
    with c2:
        default_factors = ["Technical (T)", "Economic (E)", "Environmental (En)", "Social (S)", "Governance (G)"]
        factor_text = st.text_area(
            "Factor names (one per line)",
            value="\n".join(default_factors if use_sample_bwm else ["F1", "F2", "F3"]),
            height=130,
        )

    factors = [x.strip() for x in factor_text.splitlines() if x.strip()]
    if len(factors) < 2:
        st.warning("Please provide at least 2 factors.")
        st.stop()

    render_scale_table(BWM_SCALE, BWM_MEANING, "Show BWM code legend")

    bwm_sig = ("|".join(factors), n_exp_bwm, use_sample_bwm)
    if st.button("Prepare / Refresh BWM Tables", key="prep_bwm") or st.session_state.get("bwm_sig") != bwm_sig:
        st.session_state["bwm_sig"] = bwm_sig
        st.session_state["bwm_summary_df"] = init_bwm_summary(factors, n_exp_bwm, use_sample_bwm)
        for e in range(n_exp_bwm):
            best_factor = st.session_state["bwm_summary_df"].iloc[e]["Best"]
            worst_factor = st.session_state["bwm_summary_df"].iloc[e]["Worst"]
            st.session_state[f"bwm_pair_df_{e}"] = init_bwm_pairwise_df(
                factors, best_factor, worst_factor, use_sample_bwm, seed=100 + e
            )

    st.markdown("**Step A: Best and worst factor for each expert**")
    bw_summary = st.data_editor(
        st.session_state["bwm_summary_df"],
        key="bwm_summary_editor",
        use_container_width=True,
        num_rows="fixed",
        column_config={
            "Best": st.column_config.SelectboxColumn("Best", options=factors, required=True),
            "Worst": st.column_config.SelectboxColumn("Worst", options=factors, required=True),
        }
    )

    st.markdown("**Step B: Table input of fuzzy comparisons**")
    tabs = st.tabs([f"E{i+1}" for i in range(n_exp_bwm)])
    edited_pairwise_tables = []

    for e, tab in enumerate(tabs):
        with tab:
            best_factor = bw_summary.iloc[e]["Best"]
            worst_factor = bw_summary.iloc[e]["Worst"]

            if best_factor == worst_factor:
                st.error(f"Expert {e+1}: Best and worst cannot be the same.")
                pair_df = st.session_state[f"bwm_pair_df_{e}"].copy()
            else:
                base_df = st.session_state[f"bwm_pair_df_{e}"].copy()
                if list(base_df["Factor"]) != factors:
                    base_df = init_bwm_pairwise_df(factors, best_factor, worst_factor, False, seed=100 + e)

                base_df.loc[base_df["Factor"] == best_factor, "B→j"] = "EQ"
                base_df.loc[base_df["Factor"] == worst_factor, "j→W"] = "EQ"

                pair_df = st.data_editor(
                    base_df,
                    key=f"bwm_pair_editor_{e}",
                    use_container_width=True,
                    num_rows="fixed",
                    hide_index=True,
                    column_config={
                        "Factor": st.column_config.TextColumn("Factor", disabled=True),
                        "B→j": st.column_config.SelectboxColumn("B→j", options=list(BWM_SCALE.keys()), required=True),
                        "j→W": st.column_config.SelectboxColumn("j→W", options=list(BWM_SCALE.keys()), required=True),
                    }
                )
                pair_df.loc[pair_df["Factor"] == best_factor, "B→j"] = "EQ"
                pair_df.loc[pair_df["Factor"] == worst_factor, "j→W"] = "EQ"

            edited_pairwise_tables.append(pair_df)
            st.caption(f"Best = {best_factor} | Worst = {worst_factor}")

    if st.button("Compute Fuzzy BWM Weights", type="primary"):
        valid = True
        for e in range(n_exp_bwm):
            if bw_summary.iloc[e]["Best"] == bw_summary.iloc[e]["Worst"]:
                valid = False
        if not valid:
            st.error("At least one expert has identical best and worst factors.")
            st.stop()

        expert_data = []
        for e in range(n_exp_bwm):
            best_idx = factors.index(bw_summary.iloc[e]["Best"])
            worst_idx = factors.index(bw_summary.iloc[e]["Worst"])
            table = edited_pairwise_tables[e]

            bto = []
            otw = []
            for _, row in table.iterrows():
                bto.append(BWM_SCALE[str(row["B→j"])])
                otw.append(BWM_SCALE[str(row["j→W"])])

            expert_data.append({
                "best": best_idx,
                "worst": worst_idx,
                "bto": bto,
                "otw": otw,
            })

        best_counts, worst_counts = {}, {}
        for ed in expert_data:
            best_counts[ed["best"]] = best_counts.get(ed["best"], 0) + 1
            worst_counts[ed["worst"]] = worst_counts.get(ed["worst"], 0) + 1

        common_best = max(best_counts, key=best_counts.get)
        common_worst = max(worst_counts, key=worst_counts.get)

        transformed_bto = []
        transformed_otw = []

        for ed in expert_data:
            orig_bto = ed["bto"]
            divisor_b = orig_bto[common_best]
            new_bto = []
            for j in range(len(factors)):
                if j == common_best:
                    new_bto.append((1, 1, 1))
                else:
                    l = orig_bto[j][0] / max(divisor_b[2], EPS)
                    m = orig_bto[j][1] / max(divisor_b[1], EPS)
                    u = orig_bto[j][2] / max(divisor_b[0], EPS)
                    new_bto.append((l, m, u))
            transformed_bto.append(new_bto)

            orig_otw = ed["otw"]
            divisor_w = orig_otw[common_worst]
            new_otw = []
            for j in range(len(factors)):
                if j == common_worst:
                    new_otw.append((1, 1, 1))
                else:
                    l = orig_otw[j][0] / max(divisor_w[2], EPS)
                    m = orig_otw[j][1] / max(divisor_w[1], EPS)
                    u = orig_otw[j][2] / max(divisor_w[0], EPS)
                    new_otw.append((l, m, u))
            transformed_otw.append(new_otw)

        agg_best = []
        agg_worst = []
        for j in range(len(factors)):
            agg_best.append(geometric_mean([t[j] for t in transformed_bto]))
            agg_worst.append(geometric_mean([t[j] for t in transformed_otw]))

        weights, xi = solve_bwm_aggregated(agg_best, agg_worst, common_best, common_worst)

        result_df = pd.DataFrame({
            "Factor": factors,
            "Weight TFN": [tfn_to_str(w) for w in weights],
            "GMI": [round(gmi(w), 6) for w in weights],
        }).sort_values("GMI", ascending=False)

        i1, i2, i3 = st.columns(3)
        with i1:
            st.metric("Common Best", factors[common_best])
        with i2:
            st.metric("Common Worst", factors[common_worst])
        with i3:
            st.metric("ξ", f"{xi:.6f}")

        st.dataframe(result_df, use_container_width=True, hide_index=True)
        st.bar_chart(result_df.set_index("Factor")[["GMI"]])

        st.session_state["bwm_weights"] = weights
        st.session_state["bwm_factors"] = factors
        st.session_state["bwm_common_best"] = common_best
        st.session_state["bwm_common_worst"] = common_worst

# ============================================================
# MODULE 3: LBWA + HYBRID
# ============================================================
elif module == "3) Fuzzy LBWA + Hybrid":
    st.markdown('<div class="section-head">Fuzzy LBWA and Hybrid Weighting</div>', unsafe_allow_html=True)

    use_bwm_factors = st.toggle("Use factors from BWM", value=True)

    if use_bwm_factors and "bwm_factors" in st.session_state:
        factors = st.session_state["bwm_factors"]
        best_idx_default = st.session_state.get("bwm_common_best", 0)
    else:
        n_f = st.number_input("Number of factors", min_value=2, value=5, step=1)
        factor_text = st.text_area(
            "Factor names (one per line)",
            value="\n".join([f"F{i+1}" for i in range(n_f)]),
            height=120,
        )
        factors = parse_names(factor_text, n_f, "F")
        best_idx_default = 0

    n_exp_lbwa = st.number_input("Number of experts", min_value=1, value=4, step=1)
    best_idx_lbwa = st.selectbox(
        "Best factor for LBWA",
        options=range(len(factors)),
        index=min(best_idx_default, len(factors)-1),
        format_func=lambda x: factors[x]
    )

    use_sample_lbwa = st.toggle("Use sample LBWA tables", value=False)

    render_scale_table(HYBRID_SCALE, HYBRID_MEANING, "Show hybrid priority code legend")

    lbwa_sig = ("|".join(factors), n_exp_lbwa, best_idx_lbwa, use_sample_lbwa)
    if st.button("Prepare / Refresh LBWA Tables", key="prep_lbwa") or st.session_state.get("lbwa_sig") != lbwa_sig:
        st.session_state["lbwa_sig"] = lbwa_sig
        for e in range(n_exp_lbwa):
            st.session_state[f"lbwa_df_{e}"] = init_lbwa_df(factors, use_sample_lbwa, seed=200 + e)

    st.markdown("**Expert level and λ tables**")
    lbwa_tabs = st.tabs([f"E{i+1}" for i in range(n_exp_lbwa)])
    lbwa_tables = []

    for e, tab in enumerate(lbwa_tabs):
        with tab:
            df = st.data_editor(
                st.session_state[f"lbwa_df_{e}"],
                key=f"lbwa_editor_{e}",
                use_container_width=True,
                num_rows="fixed",
                hide_index=True,
                column_config={
                    "Factor": st.column_config.TextColumn("Factor", disabled=True),
                    "Level": st.column_config.NumberColumn("Level", min_value=1, step=1, format="%d"),
                    "λ": st.column_config.NumberColumn("λ", min_value=0.0, step=0.1, format="%.3f"),
                }
            )
            lbwa_tables.append(df)

    c1, c2 = st.columns(2)
    with c1:
        alpha_code = st.selectbox("Priority of FBWM weight (α)", options=list(HYBRID_SCALE.keys()), index=3)
    with c2:
        beta_code = st.selectbox("Priority of FLBWA weight (β)", options=list(HYBRID_SCALE.keys()), index=3)

    alpha_tfn = HYBRID_SCALE[alpha_code]
    beta_tfn = HYBRID_SCALE[beta_code]

    if st.button("Compute LBWA and Hybrid Weights", type="primary"):
        level_data = []
        lambda_data = []
        for df in lbwa_tables:
            level_data.append([int(x) for x in df["Level"].tolist()])
            lambda_data.append([float(x) for x in df["λ"].tolist()])

        lbwa_weights = run_lbwa(factors, best_idx_lbwa, level_data, lambda_data)
        lbwa_df = pd.DataFrame({
            "Factor": factors,
            "LBWA TFN": [tfn_to_str(w) for w in lbwa_weights],
            "LBWA GMI": [round(gmi(w), 6) for w in lbwa_weights],
        }).sort_values("LBWA GMI", ascending=False)

        st.subheader("Fuzzy LBWA Weights")
        st.dataframe(lbwa_df, use_container_width=True, hide_index=True)
        st.bar_chart(lbwa_df.set_index("Factor")[["LBWA GMI"]])

        st.session_state["lbwa_weights"] = lbwa_weights

        if "bwm_weights" in st.session_state and len(st.session_state["bwm_weights"]) == len(factors):
            hybrid_weights = combine_weights(st.session_state["bwm_weights"], lbwa_weights, alpha_tfn, beta_tfn)
            hybrid_df = pd.DataFrame({
                "Factor": factors,
                "Hybrid TFN": [tfn_to_str(w) for w in hybrid_weights],
                "Hybrid GMI": [round(gmi(w), 6) for w in hybrid_weights],
            }).sort_values("Hybrid GMI", ascending=False)

            st.subheader("Hybrid Weights (FBWM + FLBWA)")
            st.dataframe(hybrid_df, use_container_width=True, hide_index=True)
            st.bar_chart(hybrid_df.set_index("Factor")[["Hybrid GMI"]])

            st.session_state["hybrid_weights"] = hybrid_weights
        else:
            st.warning("No compatible BWM weights found. Hybrid result is not computed.")
            st.session_state["hybrid_weights"] = lbwa_weights

# ============================================================
# MODULE 4: FUZZY COCO-SO
# ============================================================
elif module == "4) Fuzzy CoCoSo":
    st.markdown('<div class="section-head">Fuzzy CoCoSo with Bonferroni – Technology Ranking</div>', unsafe_allow_html=True)

    use_sample_alt = st.toggle("Use sample alternatives", value=True)
    if use_sample_alt:
        alt_names = ["TS-SS", "TS-HP", "TS-PCC", "TS-MPP"]
    else:
        alt_text = st.text_area("Alternative names (one per line)", value="A1\nA2\nA3", height=120)
        alt_names = [x.strip() for x in alt_text.splitlines() if x.strip()]

    if len(alt_names) == 0:
        st.warning("Please define at least one alternative.")
        st.stop()

    use_existing_weights = "hybrid_weights" in st.session_state
    if use_existing_weights:
        criteria_names = st.session_state.get("bwm_factors", [f"C{i+1}" for i in range(len(st.session_state["hybrid_weights"]))])
        init_weights = st.session_state["hybrid_weights"]
        st.info("Using hybrid weights from previous module.")
    else:
        n_crit_manual = st.number_input("Number of criteria", min_value=1, value=5, step=1)
        crit_text = st.text_area(
            "Criterion names (one per line)",
            value="\n".join([f"C{i+1}" for i in range(n_crit_manual)]),
            height=120,
        )
        criteria_names = parse_names(crit_text, n_crit_manual, "C")
        init_weights = None
        st.warning("No hybrid weights found. Please input weights manually below.")

    cocoso_sig = ("|".join(criteria_names), "|".join(alt_names), use_existing_weights)
    if st.button("Prepare / Refresh CoCoSo Tables", key="prep_cocoso") or st.session_state.get("cocoso_sig") != cocoso_sig:
        st.session_state["cocoso_sig"] = cocoso_sig
        st.session_state["cocoso_criteria_df"] = init_criteria_weight_df(criteria_names, init_weights)
        for a, alt in enumerate(alt_names):
            st.session_state[f"cocoso_alt_df_{a}"] = init_decision_df(criteria_names, use_sample=False, seed=300 + a)

    criteria_df = st.session_state["cocoso_criteria_df"].copy()
    criteria_df["Criterion"] = criteria_names

    st.markdown("**Criteria weights and types**")
    if use_existing_weights:
        edited_criteria = st.data_editor(
            criteria_df,
            key="cocoso_criteria_editor",
            use_container_width=True,
            num_rows="fixed",
            hide_index=True,
            column_config={
                "Criterion": st.column_config.TextColumn("Criterion", disabled=True),
                "w_l": st.column_config.NumberColumn("w_l", disabled=True, format="%.6f"),
                "w_m": st.column_config.NumberColumn("w_m", disabled=True, format="%.6f"),
                "w_u": st.column_config.NumberColumn("w_u", disabled=True, format="%.6f"),
                "Type": st.column_config.SelectboxColumn("Type", options=["benefit", "cost"], required=True),
            }
        )
    else:
        edited_criteria = st.data_editor(
            criteria_df,
            key="cocoso_criteria_editor_manual",
            use_container_width=True,
            num_rows="fixed",
            hide_index=True,
            column_config={
                "Criterion": st.column_config.TextColumn("Criterion", disabled=True),
                "w_l": st.column_config.NumberColumn("w_l", min_value=0.0, step=0.01, format="%.6f"),
                "w_m": st.column_config.NumberColumn("w_m", min_value=0.0, step=0.01, format="%.6f"),
                "w_u": st.column_config.NumberColumn("w_u", min_value=0.0, step=0.01, format="%.6f"),
                "Type": st.column_config.SelectboxColumn("Type", options=["benefit", "cost"], required=True),
            }
        )

    st.markdown("**Decision matrix by alternative**")
    use_sample_dm = st.toggle("Use sample decision matrices", value=False)
    if use_sample_dm:
        for a in range(len(alt_names)):
            st.session_state[f"cocoso_alt_df_{a}"] = init_decision_df(criteria_names, use_sample=True, seed=300 + a)

    alt_tabs = st.tabs(alt_names)
    edited_alt_tables = []

    for a, tab in enumerate(alt_tabs):
        with tab:
            df = st.session_state[f"cocoso_alt_df_{a}"].copy()
            df["Criterion"] = criteria_names
            edited_df = st.data_editor(
                df,
                key=f"cocoso_alt_editor_{a}",
                use_container_width=True,
                num_rows="fixed",
                hide_index=True,
                column_config={
                    "Criterion": st.column_config.TextColumn("Criterion", disabled=True),
                    "l": st.column_config.NumberColumn("l", step=0.01, format="%.6f"),
                    "m": st.column_config.NumberColumn("m", step=0.01, format="%.6f"),
                    "u": st.column_config.NumberColumn("u", step=0.01, format="%.6f"),
                }
            )
            edited_alt_tables.append(edited_df)

    p1, p2, p3 = st.columns(3)
    with p1:
        phi1 = st.number_input("ϕ1", value=1.0, step=0.1)
    with p2:
        phi2 = st.number_input("ϕ2", value=1.0, step=0.1)
    with p3:
        pi = st.number_input("π", min_value=0.0, max_value=1.0, value=0.5, step=0.05)

    if st.button("Run Fuzzy CoCoSo", type="primary"):
        if phi1 + phi2 <= 0:
            st.error("ϕ1 + ϕ2 must be greater than 0.")
            st.stop()

        weights = list(zip(
            edited_criteria["w_l"].astype(float),
            edited_criteria["w_m"].astype(float),
            edited_criteria["w_u"].astype(float),
        ))
        types = edited_criteria["Type"].astype(str).tolist()

        decision = []
        for df in edited_alt_tables:
            alt_row = []
            for _, row in df.iterrows():
                l = float(row["l"])
                m = float(row["m"])
                u = float(row["u"])
                if not (l <= m <= u):
                    st.error("Each TFN must satisfy l ≤ m ≤ u.")
                    st.stop()
                alt_row.append((l, m, u))
            decision.append(alt_row)

        norm_matrix = normalize_cocoso(decision, types)
        scob, pcob = compute_bonferroni(norm_matrix, weights, phi1, phi2)
        psi_a, psi_b, psi_c = relative_significance(scob, pcob, pi)
        final = final_scores(psi_a, psi_b, psi_c)

        ranking_df = pd.DataFrame({
            "Alternative": alt_names,
            "Final TFN": [tfn_to_str(x) for x in final],
            "GMI": [round(gmi(x), 6) for x in final],
        }).sort_values("GMI", ascending=False).reset_index(drop=True)
        ranking_df.insert(0, "Rank", range(1, len(ranking_df) + 1))

        st.subheader("Ranking Results")
        st.dataframe(ranking_df, use_container_width=True, hide_index=True)
        st.bar_chart(ranking_df.set_index("Alternative")[["GMI"]])

        st.session_state["cocoso_ranking"] = ranking_df
