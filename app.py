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
# GENERAL HELPERS
# ============================================================
def gmi(tfn):
    return (tfn[0] + 4 * tfn[1] + tfn[2]) / 6

def defuzz_tfn(tfn):
    return (tfn[0] + 4 * tfn[1] + tfn[2]) / 6.0

def safe_pos(x: float, eps: float = EPS) -> float:
    return max(float(x), eps)

def safe_normalize_to_1(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    s = np.nansum(v)
    if len(v) == 0:
        raise ValueError("Vector length cannot be zero.")
    if s <= 0 or np.isclose(s, 0.0) or np.isnan(s):
        return np.ones_like(v) / len(v)
    return v / s

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

def make_bar_df(items, scores, item_col="Item", score_col="Value"):
    return pd.DataFrame({item_col: items, score_col: scores}).set_index(item_col)

def to_bc_label(x: str) -> str:
    s = str(x).strip().upper()
    if s in {"B", "BENEFIT", "MAX"}:
        return "B"
    return "C"

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
    x0.append(0.5)

    cons = []

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

    def gmi_sum(x):
        return sum((x[i*3] + 4*x[i*3+1] + x[i*3+2]) / 6 for i in range(n)) - 1
    cons.append({'type': 'eq', 'fun': gmi_sum})

    def m_sum(x):
        return sum(x[i*3+1] for i in range(n)) - 1
    cons.append({'type': 'eq', 'fun': m_sum})

    for j in range(n):
        def l_plus_u(x, j=j):
            s = x[j*3]
            for i in range(n):
                if i != j:
                    s += x[i*3+2]
            return s - 1
        cons.append({'type': 'ineq', 'fun': l_plus_u})

    for j in range(n):
        def u_plus_l(x, j=j):
            s = x[j*3+2]
            for i in range(n):
                if i != j:
                    s += x[i*3]
            return 1 - s
        cons.append({'type': 'ineq', 'fun': u_plus_l})

    for j in range(n):
        cons.append({'type': 'ineq', 'fun': lambda x, j=j: x[j*3+1] - x[j*3]})
        cons.append({'type': 'ineq', 'fun': lambda x, j=j: x[j*3+2] - x[j*3+1]})

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
# BONFERRONI COCO-SO HELPERS
# ============================================================
def fuzzy_df_to_nested_matrix(fuzzy_df: pd.DataFrame, criteria, alternatives):
    matrix = []
    for alt in alternatives:
        row = []
        for c in criteria:
            trip = (
                float(fuzzy_df.loc[alt, f"{c}_l"]),
                float(fuzzy_df.loc[alt, f"{c}_m"]),
                float(fuzzy_df.loc[alt, f"{c}_u"]),
            )
            row.append(tuple(sorted(trip)))
        matrix.append(row)
    return matrix

def normalize_cocoso_bonferroni(decision, types_bc):
    n_alt = len(decision)
    n_crit = len(types_bc)
    norm = [[(0.0, 0.0, 0.0) for _ in range(n_crit)] for _ in range(n_alt)]

    for j in range(n_crit):
        typ = to_bc_label(types_bc[j])

        if typ == "B":
            max_u = max(decision[i][j][2] for i in range(n_alt))
            max_u = safe_pos(max_u)
            for i in range(n_alt):
                l, m, u = decision[i][j]
                norm[i][j] = (l / max_u, m / max_u, u / max_u)
        else:
            min_l = min(decision[i][j][0] for i in range(n_alt))
            min_l = safe_pos(min_l)
            for i in range(n_alt):
                l, m, u = decision[i][j]
                l = safe_pos(l)
                m = safe_pos(m)
                u = safe_pos(u)
                norm[i][j] = (min_l / u, min_l / m, min_l / l)

    return norm

def compute_bonferroni_excel_style(norm_matrix, weights, phi1=1.0, phi2=1.0):
    weights = safe_normalize_to_1(pd.Series(weights).astype(float).values)

    n_alt = len(norm_matrix)
    n_crit = len(weights)

    if n_crit < 2:
        raise ValueError("At least two criteria are required for fuzzy Bonferroni CoCoSo.")

    scob = []
    pcob = []
    exp_term = 1.0 / safe_pos(phi1 + phi2)

    for a in range(n_alt):
        s_l = 0.0
        s_m = 0.0
        s_u = 0.0

        log_p_l = 0.0
        log_p_m = 0.0
        log_p_u = 0.0

        for i in range(n_crit):
            wi = min(max(weights[i], EPS), 1.0 - EPS)
            denom = 1.0 - wi

            for j in range(n_crit):
                if i == j:
                    continue

                wj = weights[j]
                term = (wi * wj) / denom

                gi_l, gi_m, gi_u = norm_matrix[a][i]
                gj_l, gj_m, gj_u = norm_matrix[a][j]

                s_l += term * (safe_pos(gi_l) ** phi1) * (safe_pos(gj_l) ** phi2)
                s_m += term * (safe_pos(gi_m) ** phi1) * (safe_pos(gj_m) ** phi2)
                s_u += term * (safe_pos(gi_u) ** phi1) * (safe_pos(gj_u) ** phi2)

                base_l = safe_pos(phi1 * gi_l + phi2 * gj_l)
                base_m = safe_pos(phi1 * gi_m + phi2 * gj_m)
                base_u = safe_pos(phi1 * gi_u + phi2 * gj_u)

                log_p_l += term * math.log(base_l)
                log_p_m += term * math.log(base_m)
                log_p_u += term * math.log(base_u)

        s_l = safe_pos(s_l) ** exp_term
        s_m = safe_pos(s_m) ** exp_term
        s_u = safe_pos(s_u) ** exp_term
        scob.append((s_l, s_m, s_u))

        p_l = math.exp(log_p_l) / safe_pos(phi1 + phi2)
        p_m = math.exp(log_p_m) / safe_pos(phi1 + phi2)
        p_u = math.exp(log_p_u) / safe_pos(phi1 + phi2)
        pcob.append((p_l, p_m, p_u))

    return scob, pcob

def relative_significance_excel_style(scob, pcob, pi=0.5):
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

        a_l = (s[0] + p[0]) / safe_pos(sum_scob_u + sum_pcob_u)
        a_m = (s[1] + p[1]) / safe_pos(sum_scob_m + sum_pcob_m)
        a_u = (s[2] + p[2]) / safe_pos(sum_scob_l + sum_pcob_l)
        psi_a.append((a_l, a_m, a_u))

        b_l = (s[0] / safe_pos(min_scob_u)) + (p[0] / safe_pos(min_pcob_u))
        b_m = (s[1] / safe_pos(min_scob_m)) + (p[1] / safe_pos(min_pcob_m))
        b_u = (s[2] / safe_pos(min_scob_l)) + (p[2] / safe_pos(min_pcob_l))
        psi_b.append((b_l, b_m, b_u))

        c_l = (pi * s[0] + (1 - pi) * p[0]) / safe_pos(pi * max_scob_u + (1 - pi) * max_pcob_u)
        c_m = (pi * s[1] + (1 - pi) * p[1]) / safe_pos(pi * max_scob_m + (1 - pi) * max_pcob_m)
        c_u = (pi * s[2] + (1 - pi) * p[2]) / safe_pos(pi * max_scob_l + (1 - pi) * max_pcob_l)
        psi_c.append((c_l, c_m, c_u))

    return psi_a, psi_b, psi_c

def final_scores_bonferroni(psi_a, psi_b, psi_c, alternative_names=None):
    n_alt = len(psi_a)
    if alternative_names is None:
        alternative_names = [f"A{i+1}" for i in range(n_alt)]

    rows = []
    for i in range(n_alt):
        a = psi_a[i]
        b = psi_b[i]
        c = psi_c[i]

        prod_l = (safe_pos(a[0]) * safe_pos(b[0]) * safe_pos(c[0])) ** (1 / 3)
        prod_m = (safe_pos(a[1]) * safe_pos(b[1]) * safe_pos(c[1])) ** (1 / 3)
        prod_u = (safe_pos(a[2]) * safe_pos(b[2]) * safe_pos(c[2])) ** (1 / 3)

        avg_l = (a[0] + b[0] + c[0]) / 3.0
        avg_m = (a[1] + b[1] + c[1]) / 3.0
        avg_u = (a[2] + b[2] + c[2]) / 3.0

        final_tfn = (prod_l + avg_l, prod_m + avg_m, prod_u + avg_u)
        crisp = defuzz_tfn(final_tfn)

        rows.append([
            alternative_names[i],
            a[0], a[1], a[2],
            b[0], b[1], b[2],
            c[0], c[1], c[2],
            final_tfn[0], final_tfn[1], final_tfn[2],
            crisp,
        ])

    df = pd.DataFrame(
        rows,
        columns=[
            "Alternative",
            "psi_a_l", "psi_a_m", "psi_a_u",
            "psi_b_l", "psi_b_m", "psi_b_u",
            "psi_c_l", "psi_c_m", "psi_c_u",
            "Final_l", "Final_m", "Final_u",
            "Crisp",
        ],
    )
    df["Rank"] = df["Crisp"].rank(ascending=False, method="min").astype(int)
    return df.sort_values(["Crisp", "Alternative"], ascending=[False, True]).reset_index(drop=True)

def cocoso_bonferroni_from_app(fuzzy_df, types_bc, crisp_weights, phi1=1.0, phi2=1.0, pi=0.5):
    criteria = [c[:-2] for c in fuzzy_df.columns if c.endswith("_l")]
    alternatives = fuzzy_df.index.astype(str).tolist()

    decision = fuzzy_df_to_nested_matrix(fuzzy_df, criteria, alternatives)
    norm_matrix = normalize_cocoso_bonferroni(decision, types_bc)
    weights = pd.Series(crisp_weights, index=criteria).astype(float)

    scob, pcob = compute_bonferroni_excel_style(norm_matrix, weights, phi1=phi1, phi2=phi2)
    psi_a, psi_b, psi_c = relative_significance_excel_style(scob, pcob, pi=pi)
    ranking_df = final_scores_bonferroni(psi_a, psi_b, psi_c, alternative_names=alternatives)

    norm_rows = []
    for i, alt in enumerate(alternatives):
        row = {"Alternative": alt}
        for j, c in enumerate(criteria):
            row[f"{c}_l"] = norm_matrix[i][j][0]
            row[f"{c}_m"] = norm_matrix[i][j][1]
            row[f"{c}_u"] = norm_matrix[i][j][2]
        norm_rows.append(row)
    norm_df = pd.DataFrame(norm_rows).set_index("Alternative")

    scob_df = pd.DataFrame(scob, columns=["SCoB_l", "SCoB_m", "SCoB_u"], index=alternatives)
    pcob_df = pd.DataFrame(pcob, columns=["PCoB_l", "PCoB_m", "PCoB_u"], index=alternatives)
    psi_a_df = pd.DataFrame(psi_a, columns=["psi_a_l", "psi_a_m", "psi_a_u"], index=alternatives)
    psi_b_df = pd.DataFrame(psi_b, columns=["psi_b_l", "psi_b_m", "psi_b_u"], index=alternatives)
    psi_c_df = pd.DataFrame(psi_c, columns=["psi_c_l", "psi_c_m", "psi_c_u"], index=alternatives)

    meta = {"phi1": phi1, "phi2": phi2, "pi": pi}

    return ranking_df, meta, norm_df, scob_df, pcob_df, psi_a_df, psi_b_df, psi_c_df

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
        "4) Fuzzy Bonferroni CoCoSo",
    ],
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    **Workflow**
    
    1. Fuzzy Delphi  
    2. Fuzzy BWM  
    3. Fuzzy LBWA + Hybrid  
    4. Fuzzy Bonferroni CoCoSo  
    """
)
st.sidebar.caption("Major data-entry sections use compact table editors.")

# ============================================================
# HEADER
# ============================================================
st.title("Integrated Fuzzy MCDM Platform")
st.caption("Cleaner interface for Fuzzy Delphi, Fuzzy BWM, Fuzzy LBWA-Hybrid, and Fuzzy Bonferroni CoCoSo")

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
                Fuzzy Delphi uses compact abbreviation-based tables such as
                <code>VLR</code>, <code>LR</code>, <code>MR</code>, <code>HR</code>, <code>VHR</code>.
                BWM and CoCoSo also use cleaner table-based inputs.
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
        st.metric("Stage 4", "Bonferroni CoCoSo")

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
# MODULE 4: FUZZY BONFERRONNI COCO-SO
# ============================================================
elif module == "4) Fuzzy Bonferroni CoCoSo":
    st.markdown('<div class="section-head">Fuzzy Bonferroni CoCoSo – Technology Ranking</div>', unsafe_allow_html=True)

    st.info("This module uses the Bonferroni CoCoSo logic from your other app. Criterion weights are used as crisp weights after GMI defuzzification of the fuzzy weights.")

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
        criteria_names = st.session_state.get(
            "bwm_factors",
            [f"C{i+1}" for i in range(len(st.session_state["hybrid_weights"]))]
        )
        init_weights = st.session_state["hybrid_weights"]
        st.success("Hybrid fuzzy weights found from previous module.")
    else:
        n_crit_manual = st.number_input("Number of criteria", min_value=2, value=5, step=1)
        crit_text = st.text_area(
            "Criterion names (one per line)",
            value="\n".join([f"C{i+1}" for i in range(n_crit_manual)]),
            height=120,
        )
        criteria_names = parse_names(crit_text, n_crit_manual, "C")
        init_weights = None
        st.warning("No hybrid weights found. Please input fuzzy weights manually below.")

    cocoso_sig = ("|".join(criteria_names), "|".join(alt_names), use_existing_weights)
    if st.button("Prepare / Refresh Bonferroni CoCoSo Tables", key="prep_cocoso") or st.session_state.get("cocoso_sig") != cocoso_sig:
        st.session_state["cocoso_sig"] = cocoso_sig
        st.session_state["cocoso_criteria_df"] = init_criteria_weight_df(criteria_names, init_weights)
        for a, alt in enumerate(alt_names):
            st.session_state[f"cocoso_alt_df_{a}"] = init_decision_df(criteria_names, use_sample=False, seed=300 + a)

    criteria_df = st.session_state["cocoso_criteria_df"].copy()
    criteria_df["Criterion"] = criteria_names

    st.markdown("**Criteria fuzzy weights and criterion type**")
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

    crisp_preview = []
    for _, row in edited_criteria.iterrows():
        crisp_preview.append(defuzz_tfn((float(row["w_l"]), float(row["w_m"]), float(row["w_u"]))))

    preview_df = pd.DataFrame({
        "Criterion": edited_criteria["Criterion"].tolist(),
        "Crisp weight (GMI)": crisp_preview
    })
    st.caption("These crisp weights are used inside Bonferroni CoCoSo after GMI defuzzification.")
    st.dataframe(preview_df, use_container_width=True, hide_index=True)

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

    if st.button("Run Fuzzy Bonferroni CoCoSo", type="primary"):
        if len(criteria_names) < 2:
            st.error("At least two criteria are required.")
            st.stop()

        if phi1 + phi2 <= 0:
            st.error("ϕ1 + ϕ2 must be greater than 0.")
            st.stop()

        fuzzy_weights = list(zip(
            edited_criteria["w_l"].astype(float),
            edited_criteria["w_m"].astype(float),
            edited_criteria["w_u"].astype(float),
        ))
        crisp_weights = [defuzz_tfn(w) for w in fuzzy_weights]
        types = [to_bc_label(t) for t in edited_criteria["Type"].astype(str).tolist()]

        fuzzy_rows = []
        for a, df in enumerate(edited_alt_tables):
            row_dict = {}
            for _, row in df.iterrows():
                c = str(row["Criterion"])
                l = float(row["l"])
                m = float(row["m"])
                u = float(row["u"])
                if not (l <= m <= u):
                    st.error(f"Invalid TFN in {alt_names[a]} for {c}: must satisfy l ≤ m ≤ u.")
                    st.stop()
                row_dict[f"{c}_l"] = l
                row_dict[f"{c}_m"] = m
                row_dict[f"{c}_u"] = u
            fuzzy_rows.append(row_dict)

        fuzzy_df = pd.DataFrame(fuzzy_rows, index=alt_names)
        fuzzy_df = fuzzy_df[[col for c in criteria_names for col in [f"{c}_l", f"{c}_m", f"{c}_u"]]]

        ranking_df, meta, norm_df, scob_df, pcob_df, psi_a_df, psi_b_df, psi_c_df = cocoso_bonferroni_from_app(
            fuzzy_df=fuzzy_df,
            types_bc=types,
            crisp_weights=crisp_weights,
            phi1=float(phi1),
            phi2=float(phi2),
            pi=float(pi),
        )

        st.subheader("Final Ranking")
        show_rank_df = ranking_df[["Rank", "Alternative", "Final_l", "Final_m", "Final_u", "Crisp"]].copy()
        st.dataframe(show_rank_df, use_container_width=True, hide_index=True)
        st.bar_chart(ranking_df.set_index("Alternative")[["Crisp"]])

        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("ϕ1", f"{meta['phi1']:.3f}")
        with m2:
            st.metric("ϕ2", f"{meta['phi2']:.3f}")
        with m3:
            st.metric("π", f"{meta['pi']:.3f}")

        with st.expander("Normalized Bonferroni matrix", expanded=False):
            st.dataframe(norm_df, use_container_width=True)

        with st.expander("SCoB", expanded=False):
            st.dataframe(scob_df, use_container_width=True)

        with st.expander("PCoB", expanded=False):
            st.dataframe(pcob_df, use_container_width=True)

        with st.expander("psi_a", expanded=False):
            st.dataframe(psi_a_df, use_container_width=True)

        with st.expander("psi_b", expanded=False):
            st.dataframe(psi_b_df, use_container_width=True)

        with st.expander("psi_c", expanded=False):
            st.dataframe(psi_c_df, use_container_width=True)

        st.session_state["bonferroni_ranking"] = ranking_df
        st.session_state["bonferroni_norm_df"] = norm_df
        st.session_state["bonferroni_scob_df"] = scob_df
        st.session_state["bonferroni_pcob_df"] = pcob_df
        st.session_state["bonferroni_psi_a_df"] = psi_a_df
        st.session_state["bonferroni_psi_b_df"] = psi_b_df
        st.session_state["bonferroni_psi_c_df"] = psi_c_df
