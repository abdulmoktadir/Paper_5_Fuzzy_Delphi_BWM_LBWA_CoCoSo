import math
import numpy as np
import pandas as pd
import streamlit as st
from scipy.optimize import linprog

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
.top-factor {
    padding: 0.9rem 1rem;
    border-radius: 14px;
    background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%);
    border: 1px solid #86efac;
    color: #14532d;
    font-weight: 700;
    margin-top: 0.4rem;
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
    return (tfn[0] + 4 * tfn[1] + tfn[2]) / 6.0

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

def tfn_div(a, b):
    return (
        a[0] / max(b[2], EPS),
        a[1] / max(b[1], EPS),
        a[2] / max(b[0], EPS),
    )

def geometric_mean(tfns):
    prod_l, prod_m, prod_u = 1.0, 1.0, 1.0
    n = len(tfns)
    for t in tfns:
        prod_l *= max(t[0], EPS)
        prod_m *= max(t[1], EPS)
        prod_u *= max(t[2], EPS)
    return (prod_l ** (1 / n), prod_m ** (1 / n), prod_u ** (1 / n))

def tfn_to_str(t, digits=3):
    return f"({t[0]:.{digits}f}, {t[1]:.{digits}f}, {t[2]:.{digits}f})"

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
    return "B" if s in {"B", "BENEFIT", "MAX"} else "C"

def highlight_top_factor(row):
    if row["Rank"] == 1:
        return ["background-color: #dcfce7; font-weight: 700; color: #14532d;"] * len(row)
    return [""] * len(row)

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
# FUZZY BWM: TRANSFORMATION + AGGREGATION
# ============================================================
def choose_common_factor(idxs):
    counts = {}
    for idx in idxs:
        counts[idx] = counts.get(idx, 0) + 1
    max_count = max(counts.values())
    winners = [k for k, v in counts.items() if v == max_count]
    return min(winners), counts

def transform_bwm_expert(expert, common_best_idx, common_worst_idx, n):
    orig_bto = expert["bto"]
    orig_otw = expert["otw"]

    divisor_b = orig_bto[common_best_idx]
    divisor_w = orig_otw[common_worst_idx]

    transformed_bto = []
    transformed_otw = []

    for j in range(n):
        if j == common_best_idx:
            transformed_bto.append((1, 1, 1))
        else:
            transformed_bto.append(tfn_div(orig_bto[j], divisor_b))

    for j in range(n):
        if j == common_worst_idx:
            transformed_otw.append((1, 1, 1))
        else:
            transformed_otw.append(tfn_div(orig_otw[j], divisor_w))

    return transformed_bto, transformed_otw

def aggregate_bwm_vectors(transformed_experts, n):
    agg_best = []
    agg_worst = []
    for j in range(n):
        agg_best.append(geometric_mean([ex["bto_trans"][j] for ex in transformed_experts]))
        agg_worst.append(geometric_mean([ex["otw_trans"][j] for ex in transformed_experts]))
    return agg_best, agg_worst

def build_vector_table(expert_list, factors, key_name, expert_labels, digits=3):
    rows = []
    for e_label, ex in zip(expert_labels, expert_list):
        row = {"Expert": e_label}
        for j, f in enumerate(factors):
            row[f] = tfn_to_str(ex[key_name][j], digits)
        rows.append(row)
    return pd.DataFrame(rows)

def build_aggregate_table(factors, agg_best, agg_worst, digits=3):
    return pd.DataFrame({
        "Factor": factors,
        "Aggregated Converted f_B→other": [tfn_to_str(x, digits) for x in agg_best],
        "Aggregated Converted other→f_W": [tfn_to_str(x, digits) for x in agg_worst],
    })

# ============================================================
# FUZZY BWM: DIRECT LP SOLVER
# ============================================================
def solve_bwm_aggregated_lp(agg_best, agg_worst, best_idx, worst_idx):
    n = len(agg_best)
    nvar = 3 * n + 1
    xi_idx = 3 * n

    def idx_l(i): return 3 * i
    def idx_m(i): return 3 * i + 1
    def idx_u(i): return 3 * i + 2

    A_ub, b_ub = [], []
    A_eq, b_eq = [], []

    for j in range(n):
        if j == best_idx:
            continue

        lBj, mBj, uBj = agg_best[j]

        row = np.zeros(nvar)
        row[idx_l(best_idx)] = 1
        row[idx_l(j)] = -uBj
        row[xi_idx] = -1
        A_ub.append(row)
        b_ub.append(0)

        row = np.zeros(nvar)
        row[idx_l(best_idx)] = -1
        row[idx_l(j)] = uBj
        row[xi_idx] = -1
        A_ub.append(row)
        b_ub.append(0)

        row = np.zeros(nvar)
        row[idx_m(best_idx)] = 1
        row[idx_m(j)] = -mBj
        row[xi_idx] = -1
        A_ub.append(row)
        b_ub.append(0)

        row = np.zeros(nvar)
        row[idx_m(best_idx)] = -1
        row[idx_m(j)] = mBj
        row[xi_idx] = -1
        A_ub.append(row)
        b_ub.append(0)

        row = np.zeros(nvar)
        row[idx_u(best_idx)] = 1
        row[idx_u(j)] = -lBj
        row[xi_idx] = -1
        A_ub.append(row)
        b_ub.append(0)

        row = np.zeros(nvar)
        row[idx_u(best_idx)] = -1
        row[idx_u(j)] = lBj
        row[xi_idx] = -1
        A_ub.append(row)
        b_ub.append(0)

    for j in range(n):
        if j == worst_idx:
            continue

        ljW, mjW, ujW = agg_worst[j]

        row = np.zeros(nvar)
        row[idx_l(j)] = 1
        row[idx_l(worst_idx)] = -ujW
        row[xi_idx] = -1
        A_ub.append(row)
        b_ub.append(0)

        row = np.zeros(nvar)
        row[idx_l(j)] = -1
        row[idx_l(worst_idx)] = ujW
        row[xi_idx] = -1
        A_ub.append(row)
        b_ub.append(0)

        row = np.zeros(nvar)
        row[idx_m(j)] = 1
        row[idx_m(worst_idx)] = -mjW
        row[xi_idx] = -1
        A_ub.append(row)
        b_ub.append(0)

        row = np.zeros(nvar)
        row[idx_m(j)] = -1
        row[idx_m(worst_idx)] = mjW
        row[xi_idx] = -1
        A_ub.append(row)
        b_ub.append(0)

        row = np.zeros(nvar)
        row[idx_u(j)] = 1
        row[idx_u(worst_idx)] = -ljW
        row[xi_idx] = -1
        A_ub.append(row)
        b_ub.append(0)

        row = np.zeros(nvar)
        row[idx_u(j)] = -1
        row[idx_u(worst_idx)] = ljW
        row[xi_idx] = -1
        A_ub.append(row)
        b_ub.append(0)

    row = np.zeros(nvar)
    for i in range(n):
        row[idx_l(i)] = 1 / 6
        row[idx_m(i)] = 4 / 6
        row[idx_u(i)] = 1 / 6
    A_eq.append(row)
    b_eq.append(1.0)

    row = np.zeros(nvar)
    for i in range(n):
        row[idx_m(i)] = 1
    A_eq.append(row)
    b_eq.append(1.0)

    for j in range(n):
        row = np.zeros(nvar)
        row[idx_u(j)] = 1
        for i in range(n):
            if i != j:
                row[idx_l(i)] = 1
        A_ub.append(row)
        b_ub.append(1.0)

    for j in range(n):
        row = np.zeros(nvar)
        row[idx_l(j)] = -1
        for i in range(n):
            if i != j:
                row[idx_u(i)] = -1
        A_ub.append(row)
        b_ub.append(-1.0)

    for i in range(n):
        row = np.zeros(nvar)
        row[idx_l(i)] = 1
        row[idx_m(i)] = -1
        A_ub.append(row)
        b_ub.append(0)

        row = np.zeros(nvar)
        row[idx_m(i)] = 1
        row[idx_u(i)] = -1
        A_ub.append(row)
        b_ub.append(0)

    c = np.zeros(nvar)
    c[xi_idx] = 1.0
    bounds = [(0, None)] * (3 * n) + [(0, 1)]

    res = linprog(
        c=c,
        A_ub=np.array(A_ub, dtype=float),
        b_ub=np.array(b_ub, dtype=float),
        A_eq=np.array(A_eq, dtype=float),
        b_eq=np.array(b_eq, dtype=float),
        bounds=bounds,
        method="highs",
    )

    if not res.success:
        raise RuntimeError("Linear optimization failed: " + res.message)

    x = res.x
    weights = [(x[idx_l(i)], x[idx_m(i)], x[idx_u(i)]) for i in range(n)]
    xi = x[xi_idx]
    return weights, xi

# ============================================================
# FUZZY LBWA (SINGLE-TABLE INPUT, EXCEL-ALIGNED)
# ============================================================
def scalar_divide_tfn(a, tfn_matrix):
    return np.column_stack([
        a / np.maximum(tfn_matrix[:, 2], EPS),
        a / np.maximum(tfn_matrix[:, 1], EPS),
        a / np.maximum(tfn_matrix[:, 0], EPS),
    ])

def defuzzify_weighted(tfn_matrix):
    return (tfn_matrix[:, 0] + 4 * tfn_matrix[:, 1] + tfn_matrix[:, 2]) / 6.0

def run_lbwa_excel_single_table(input_df, num_experts, theta, reference_idx):
    expected_cols = ["Factor", "Qi"] + [f"E{i+1}" for i in range(num_experts)]
    missing_cols = [c for c in expected_cols if c not in input_df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    work_df = input_df.copy()
    work_df["Factor"] = work_df["Factor"].astype(str).str.strip()
    work_df["Factor"] = [
        name if name else f"Factor {i+1}"
        for i, name in enumerate(work_df["Factor"])
    ]

    work_df["Qi"] = pd.to_numeric(work_df["Qi"], errors="coerce")
    expert_cols = [f"E{i+1}" for i in range(num_experts)]
    for c in expert_cols:
        work_df[c] = pd.to_numeric(work_df[c], errors="coerce")

    if work_df["Qi"].isna().any():
        raise ValueError("Qi contains invalid or empty values.")
    if work_df[expert_cols].isna().any().any():
        raise ValueError("One or more expert score cells contain invalid or empty values.")

    qi_arr = work_df["Qi"].astype(float).values
    data = work_df[expert_cols].astype(float).values

    if np.any(qi_arr < 0):
        raise ValueError("Qi values must be non-negative.")
    if np.any(data < 0):
        raise ValueError("Expert scores must be non-negative.")
    if theta <= 0:
        raise ValueError("Theta (θ) must be greater than zero.")

    # TFN = (min, mean, max) across expert columns
    tfn = np.column_stack([
        np.min(data, axis=1),
        np.mean(data, axis=1),
        np.max(data, axis=1)
    ])

    tfn_df = pd.DataFrame(tfn, columns=["l", "m", "u"])
    tfn_df.insert(0, "Qi", qi_arr)
    tfn_df.insert(0, "Factor", work_df["Factor"])

    # denominator = (Qi*theta + l, Qi*theta + m, Qi*theta + u)
    denominator_tfn = np.column_stack([
        qi_arr * theta + tfn[:, 0],
        qi_arr * theta + tfn[:, 1],
        qi_arr * theta + tfn[:, 2]
    ])

    # influence = theta / denominator with reversed TFN division
    influence = scalar_divide_tfn(theta, denominator_tfn)

    influence_df = pd.DataFrame(influence, columns=["l", "m", "u"])
    influence_df.insert(0, "Qi", qi_arr)
    influence_df.insert(0, "Factor", work_df["Factor"])

    n = len(work_df)
    mask_others = np.ones(n, dtype=bool)
    mask_others[reference_idx] = False

    ref_weight = np.array([
        1 / np.maximum(1 + np.sum(influence[mask_others, 2]), EPS),
        1 / np.maximum(1 + np.sum(influence[mask_others, 1]), EPS),
        1 / np.maximum(1 + np.sum(influence[mask_others, 0]), EPS),
    ])

    fuzzy_weights = np.zeros_like(influence)
    fuzzy_weights[reference_idx] = ref_weight

    for i in range(n):
        if i != reference_idx:
            fuzzy_weights[i] = ref_weight * influence[i]

    fuzzy_weight_df = pd.DataFrame(fuzzy_weights, columns=["l", "m", "u"])
    fuzzy_weight_df.insert(0, "Factor", work_df["Factor"])

    crisp_values = defuzzify_weighted(fuzzy_weights)
    crisp_sum = np.sum(crisp_values)

    if crisp_sum <= 0:
        raise ValueError("The sum of crisp values is zero, so normalization cannot be done.")

    normalized_weights = crisp_values / crisp_sum

    result_df = pd.DataFrame({
        "Factor": work_df["Factor"],
        "Qi": qi_arr,
        "Crisp Value": crisp_values,
        "Normalized Weight": normalized_weights
    })

    result_df["Rank"] = result_df["Normalized Weight"].rank(
        ascending=False, method="dense"
    ).astype(int)

    result_df = result_df[
        ["Rank", "Factor", "Qi", "Crisp Value", "Normalized Weight"]
    ].sort_values("Normalized Weight", ascending=False).reset_index(drop=True)

    weights = [tuple(x) for x in fuzzy_weights]
    top_factor = str(result_df.iloc[0]["Factor"])
    top_weight = float(result_df.iloc[0]["Normalized Weight"])

    return {
        "input_df": work_df,
        "tfn_df": tfn_df,
        "influence_df": influence_df,
        "fuzzy_weight_df": fuzzy_weight_df,
        "result_df": result_df,
        "weights": weights,
        "top_factor": top_factor,
        "top_weight": top_weight,
        "crisp_sum": float(crisp_sum),
    }

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
        composite.append((
            num_l[i] / max(denom_u, EPS),
            num_m[i] / max(denom_m, EPS),
            num_u[i] / max(denom_l, EPS),
        ))
    return composite

# ============================================================
# BONFERRONI COCO-SO HELPERS
# ============================================================
def fuzzy_df_to_nested_matrix(fuzzy_df, criteria, alternatives):
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
    n_alt, n_crit = len(decision), len(types_bc)
    norm = [[(0.0, 0.0, 0.0) for _ in range(n_crit)] for _ in range(n_alt)]

    for j in range(n_crit):
        typ = to_bc_label(types_bc[j])

        if typ == "B":
            max_u = safe_pos(max(decision[i][j][2] for i in range(n_alt)))
            for i in range(n_alt):
                l, m, u = decision[i][j]
                norm[i][j] = (l / max_u, m / max_u, u / max_u)
        else:
            min_l = safe_pos(min(decision[i][j][0] for i in range(n_alt)))
            for i in range(n_alt):
                l, m, u = decision[i][j]
                norm[i][j] = (
                    min_l / safe_pos(u),
                    min_l / safe_pos(m),
                    min_l / safe_pos(l),
                )
    return norm

def compute_bonferroni_excel_style(norm_matrix, weights, phi1=1.0, phi2=1.0):
    weights = safe_normalize_to_1(pd.Series(weights).astype(float).values)

    n_alt, n_crit = len(norm_matrix), len(weights)
    if n_crit < 2:
        raise ValueError("At least two criteria are required for fuzzy Bonferroni CoCoSo.")

    scob, pcob = [], []
    exp_term = 1.0 / safe_pos(phi1 + phi2)

    for a in range(n_alt):
        s_l = s_m = s_u = 0.0
        log_p_l = log_p_m = log_p_u = 0.0

        for i in range(n_crit):
            wi = min(max(weights[i], EPS), 1.0 - EPS)
            denom = 1.0 - wi

            for j in range(n_crit):
                if i == j:
                    continue

                term = (wi * weights[j]) / denom

                gi_l, gi_m, gi_u = norm_matrix[a][i]
                gj_l, gj_m, gj_u = norm_matrix[a][j]

                s_l += term * (safe_pos(gi_l) ** phi1) * (safe_pos(gj_l) ** phi2)
                s_m += term * (safe_pos(gi_m) ** phi1) * (safe_pos(gj_m) ** phi2)
                s_u += term * (safe_pos(gi_u) ** phi1) * (safe_pos(gj_u) ** phi2)

                log_p_l += term * math.log(safe_pos(phi1 * gi_l + phi2 * gj_l))
                log_p_m += term * math.log(safe_pos(phi1 * gi_m + phi2 * gj_m))
                log_p_u += term * math.log(safe_pos(phi1 * gi_u + phi2 * gj_u))

        scob.append((
            safe_pos(s_l) ** exp_term,
            safe_pos(s_m) ** exp_term,
            safe_pos(s_u) ** exp_term,
        ))
        pcob.append((
            math.exp(log_p_l) / safe_pos(phi1 + phi2),
            math.exp(log_p_m) / safe_pos(phi1 + phi2),
            math.exp(log_p_u) / safe_pos(phi1 + phi2),
        ))

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

        psi_a.append((
            (s[0] + p[0]) / safe_pos(sum_scob_u + sum_pcob_u),
            (s[1] + p[1]) / safe_pos(sum_scob_m + sum_pcob_m),
            (s[2] + p[2]) / safe_pos(sum_scob_l + sum_pcob_l),
        ))

        psi_b.append((
            s[0] / safe_pos(min_scob_u) + p[0] / safe_pos(min_pcob_u),
            s[1] / safe_pos(min_scob_m) + p[1] / safe_pos(min_pcob_m),
            s[2] / safe_pos(min_scob_l) + p[2] / safe_pos(min_pcob_l),
        ))

        psi_c.append((
            (pi * s[0] + (1 - pi) * p[0]) / safe_pos(pi * max_scob_u + (1 - pi) * max_pcob_u),
            (pi * s[1] + (1 - pi) * p[1]) / safe_pos(pi * max_scob_m + (1 - pi) * max_pcob_m),
            (pi * s[2] + (1 - pi) * p[2]) / safe_pos(pi * max_scob_l + (1 - pi) * max_pcob_l),
        ))

    return psi_a, psi_b, psi_c

def final_scores_bonferroni(psi_a, psi_b, psi_c, alternative_names=None):
    n_alt = len(psi_a)
    if alternative_names is None:
        alternative_names = [f"A{i+1}" for i in range(n_alt)]

    rows = []
    for i in range(n_alt):
        a, b, c = psi_a[i], psi_b[i], psi_c[i]

        prod_l = (safe_pos(a[0]) * safe_pos(b[0]) * safe_pos(c[0])) ** (1 / 3)
        prod_m = (safe_pos(a[1]) * safe_pos(b[1]) * safe_pos(c[1])) ** (1 / 3)
        prod_u = (safe_pos(a[2]) * safe_pos(b[2]) * safe_pos(c[2])) ** (1 / 3)

        avg_l = (a[0] + b[0] + c[0]) / 3.0
        avg_m = (a[1] + b[1] + c[1]) / 3.0
        avg_u = (a[2] + b[2] + c[2]) / 3.0

        final_tfn = (prod_l + avg_l, prod_m + avg_m, prod_u + avg_u)

        rows.append([
            alternative_names[i],
            *a,
            *b,
            *c,
            *final_tfn,
            defuzz_tfn(final_tfn),
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
    scob, pcob = compute_bonferroni_excel_style(norm_matrix, crisp_weights, phi1, phi2)
    psi_a, psi_b, psi_c = relative_significance_excel_style(scob, pcob, pi)
    ranking_df = final_scores_bonferroni(psi_a, psi_b, psi_c, alternatives)

    norm_rows = []
    for i, alt in enumerate(alternatives):
        row = {"Alternative": alt}
        for j, c in enumerate(criteria):
            row[f"{c}_l"], row[f"{c}_m"], row[f"{c}_u"] = norm_matrix[i][j]
        norm_rows.append(row)

    norm_df = pd.DataFrame(norm_rows).set_index("Alternative")
    scob_df = pd.DataFrame(scob, columns=["SCoB_l", "SCoB_m", "SCoB_u"], index=alternatives)
    pcob_df = pd.DataFrame(pcob, columns=["PCoB_l", "PCoB_m", "PCoB_u"], index=alternatives)
    psi_a_df = pd.DataFrame(psi_a, columns=["psi_a_l", "psi_a_m", "psi_a_u"], index=alternatives)
    psi_b_df = pd.DataFrame(psi_b, columns=["psi_b_l", "psi_b_m", "psi_b_u"], index=alternatives)
    psi_c_df = pd.DataFrame(psi_c, columns=["psi_c_l", "psi_c_m", "psi_c_u"], index=alternatives)

    return ranking_df, {"phi1": phi1, "phi2": phi2, "pi": pi}, norm_df, scob_df, pcob_df, psi_a_df, psi_b_df, psi_c_df

# ============================================================
# SAMPLE INITIALIZERS
# ============================================================
def get_bwm_paper_sample(factors):
    target = ["T", "E", "En", "S", "G"]
    if factors != target:
        return None, None

    summary = pd.DataFrame(
        [["E", "En"], ["E", "S"], ["T", "S"], ["E", "S"], ["G", "S"]],
        columns=["Best", "Worst"],
        index=[f"E{i+1}" for i in range(5)],
    )

    tables = []
    data = [
        {"B→j": ["L", "EQ", "VH", "H", "M"],  "j→W": ["M", "VH", "EQ", "VL", "L"]},
        {"B→j": ["VL", "EQ", "H", "VH", "L"], "j→W": ["H", "VH", "L", "EQ", "M"]},
        {"B→j": ["EQ", "VL", "M", "VH", "L"], "j→W": ["VH", "H", "VL", "EQ", "VL"]},
        {"B→j": ["VL", "EQ", "H", "VH", "L"], "j→W": ["H", "VH", "VL", "EQ", "M"]},
        {"B→j": ["L", "VL", "M", "VH", "EQ"], "j→W": ["M", "H", "VL", "EQ", "VH"]},
    ]

    for d in data:
        tables.append(pd.DataFrame({
            "Factor": factors,
            "B→j": d["B→j"],
            "j→W": d["j→W"],
        }))

    return summary, tables

def init_delphi_table(criteria_names, n_exp, use_sample=False):
    rng = np.random.default_rng(42)
    expert_cols = [f"E{i+1}" for i in range(n_exp)]
    if use_sample:
        vals = rng.choice(list(DELHI_SCALE.keys()), size=(len(criteria_names), n_exp), replace=True)
        return pd.DataFrame(vals, index=criteria_names, columns=expert_cols)
    return pd.DataFrame("MR", index=criteria_names, columns=expert_cols)

def init_bwm_summary(factors, n_exp, use_sample=False):
    if use_sample and n_exp == 5:
        s, _ = get_bwm_paper_sample(factors)
        if s is not None:
            return s

    rng = np.random.default_rng(123)
    rows = []
    for _ in range(n_exp):
        if use_sample:
            best_idx = int(rng.integers(0, len(factors)))
            worst_idx = int(rng.integers(0, len(factors) - 1))
            if worst_idx >= best_idx:
                worst_idx += 1
            rows.append([factors[best_idx], factors[worst_idx]])
        else:
            rows.append([factors[0], factors[-1]])

    return pd.DataFrame(rows, columns=["Best", "Worst"], index=[f"E{i+1}" for i in range(n_exp)])

def init_bwm_pairwise_df(factors, best_factor, worst_factor, use_sample=False, seed=0, expert_idx=None):
    if use_sample and expert_idx is not None:
        _, tables = get_bwm_paper_sample(factors)
        if tables is not None and expert_idx < len(tables):
            return tables[expert_idx].copy()

    rng = np.random.default_rng(seed)
    codes = ["VL", "L", "M", "H", "VH"]
    b_codes = rng.choice(codes, size=len(factors)) if use_sample else np.array(["M"] * len(factors))
    w_codes = rng.choice(codes, size=len(factors)) if use_sample else np.array(["M"] * len(factors))

    df = pd.DataFrame({
        "Factor": factors,
        "B→j": b_codes,
        "j→W": w_codes,
    })
    df.loc[df["Factor"] == best_factor, "B→j"] = "EQ"
    df.loc[df["Factor"] == worst_factor, "j→W"] = "EQ"
    return df

def init_lbwa_editor_df(factors, n_exp, use_sample=False, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "Factor": factors,
        "Qi": rng.integers(1, 6, size=len(factors)) if use_sample else np.ones(len(factors), dtype=int),
    })

    for i in range(n_exp):
        if use_sample:
            df[f"E{i+1}"] = np.round(rng.uniform(0.0, 2.0, size=len(factors)), 4)
        else:
            df[f"E{i+1}"] = np.zeros(len(factors), dtype=float)
    return df

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
# SIDEBAR
# ============================================================
st.sidebar.title("📘 Model Navigator")
module = st.sidebar.radio(
    "Choose model",
    ["🏠 Home", "1) Fuzzy Delphi", "2) Fuzzy BWM", "3) Fuzzy LBWA + Hybrid", "4) Fuzzy Bonferroni CoCoSo"]
)
st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Workflow**\n\n"
    "1. Fuzzy Delphi  \n"
    "2. Fuzzy BWM  \n"
    "3. Fuzzy LBWA + Hybrid  \n"
    "4. Fuzzy Bonferroni CoCoSo"
)

st.title("Integrated Fuzzy MCDM Platform")
st.caption("BWM uses transformed aggregated converted vectors directly in a linear optimization model")

# ============================================================
# HOME
# ============================================================
if module == "🏠 Home":
    st.markdown('<div class="section-head">Overview</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="app-card">
            <div class="card-title">What is fixed in this version</div>
            <div class="small-note">
            The Fuzzy BWM block follows:
            initial expert vectors → common best/worst → transformation →
            aggregated converted vectors → direct optimization from TFNs.
            No defuzzification is used before solving the BWM model.
            <br><br>
            The Fuzzy LBWA block now follows the same single editable-table input style as the standalone FLBWA app:
            Factor + Qi + expert columns in one table, then Excel-aligned LBWA computation.
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

    edited_delphi = st.data_editor(
        st.session_state["delphi_df"],
        key="delphi_editor",
        use_container_width=True,
        num_rows="fixed",
        column_config={
            col: st.column_config.SelectboxColumn(col, options=list(DELHI_SCALE.keys()), required=True)
            for col in st.session_state["delphi_df"].columns
        },
    )

    if st.button("Run Fuzzy Delphi", type="primary"):
        criteria_tfns = [
            [DELHI_SCALE[str(edited_delphi.loc[crit, col])] for col in edited_delphi.columns]
            for crit in edited_delphi.index
        ]
        selected, agg_tfns, gmi_vals = run_delphi(criteria_tfns, threshold)

        results_df = pd.DataFrame({
            "Criterion": criteria_names,
            "Aggregated TFN": [tfn_to_str(x, 4) for x in agg_tfns],
            "GMI": [round(x, 4) for x in gmi_vals],
            "Selected": ["Yes" if x else "No" for x in selected],
        })
        st.dataframe(results_df, use_container_width=True, hide_index=True)
        st.bar_chart(make_bar_df(criteria_names, gmi_vals, "Criterion", "GMI"))

# ============================================================
# MODULE 2: FUZZY BWM
# ============================================================
elif module == "2) Fuzzy BWM":
    st.markdown('<div class="section-head">Fuzzy BWM – Transformation, Aggregation, and Direct LP Solution</div>', unsafe_allow_html=True)

    c1, c2 = st.columns([1, 1.4])
    with c1:
        use_sample_bwm = st.toggle("Load paper demo sample", value=True)
        n_exp_bwm = st.number_input("Number of experts", min_value=1, value=5, step=1)
    with c2:
        factor_text = st.text_area(
            "Factor names (one per line)",
            value="\n".join(["T", "E", "En", "S", "G"] if use_sample_bwm else ["F1", "F2", "F3"]),
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
                factors, best_factor, worst_factor, use_sample_bwm, seed=100 + e, expert_idx=e
            )

    st.markdown("**Step A: Expert best and worst selections**")
    bw_summary = st.data_editor(
        st.session_state["bwm_summary_df"],
        key="bwm_summary_editor",
        use_container_width=True,
        num_rows="fixed",
        column_config={
            "Best": st.column_config.SelectboxColumn("Best", options=factors, required=True),
            "Worst": st.column_config.SelectboxColumn("Worst", options=factors, required=True),
        },
    )

    st.markdown("**Step B: Initial expert fuzzy vectors**")
    expert_tabs = st.tabs([f"Ex{i+1}" for i in range(n_exp_bwm)])
    edited_pairwise_tables = []

    for e, tab in enumerate(expert_tabs):
        with tab:
            best_factor = bw_summary.iloc[e]["Best"]
            worst_factor = bw_summary.iloc[e]["Worst"]

            base_df = st.session_state[f"bwm_pair_df_{e}"].copy()
            if list(base_df["Factor"]) != factors:
                base_df = init_bwm_pairwise_df(factors, best_factor, worst_factor, False, seed=100 + e, expert_idx=None)

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
                },
            )
            pair_df.loc[pair_df["Factor"] == best_factor, "B→j"] = "EQ"
            pair_df.loc[pair_df["Factor"] == worst_factor, "j→W"] = "EQ"
            edited_pairwise_tables.append(pair_df)

            st.caption(f"Best = {best_factor} | Worst = {worst_factor}")

    if st.button("Solve Corrected Fuzzy BWM", type="primary"):
        for e in range(n_exp_bwm):
            if bw_summary.iloc[e]["Best"] == bw_summary.iloc[e]["Worst"]:
                st.error("At least one expert has identical best and worst factor.")
                st.stop()

        expert_labels = [f"Ex{i+1}" for i in range(n_exp_bwm)]
        expert_data = []

        for e in range(n_exp_bwm):
            best_idx = factors.index(bw_summary.iloc[e]["Best"])
            worst_idx = factors.index(bw_summary.iloc[e]["Worst"])
            table = edited_pairwise_tables[e]

            bto = [BWM_SCALE[str(row["B→j"])] for _, row in table.iterrows()]
            otw = [BWM_SCALE[str(row["j→W"])] for _, row in table.iterrows()]

            expert_data.append({
                "best_idx": best_idx,
                "worst_idx": worst_idx,
                "bto_init": bto,
                "otw_init": otw,
            })

        common_best_idx, best_counts = choose_common_factor([ex["best_idx"] for ex in expert_data])
        common_worst_idx, worst_counts = choose_common_factor([ex["worst_idx"] for ex in expert_data])

        transformed_experts = []
        for ex in expert_data:
            bto_trans, otw_trans = transform_bwm_expert(
                {"bto": ex["bto_init"], "otw": ex["otw_init"]},
                common_best_idx,
                common_worst_idx,
                len(factors),
            )
            transformed_experts.append({
                "best_idx": ex["best_idx"],
                "worst_idx": ex["worst_idx"],
                "bto_init": ex["bto_init"],
                "otw_init": ex["otw_init"],
                "bto_trans": bto_trans,
                "otw_trans": otw_trans,
            })

        agg_best, agg_worst = aggregate_bwm_vectors(transformed_experts, len(factors))

        try:
            weights, xi = solve_bwm_aggregated_lp(agg_best, agg_worst, common_best_idx, common_worst_idx)
        except Exception as e:
            st.error(str(e))
            st.stop()

        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Common Best", factors[common_best_idx])
        with m2:
            st.metric("Common Worst", factors[common_worst_idx])
        with m3:
            st.metric("ξ*", f"{xi:.6f}")

        count_df = pd.DataFrame({
            "Factor": factors,
            "Best count": [best_counts.get(i, 0) for i in range(len(factors))],
            "Worst count": [worst_counts.get(i, 0) for i in range(len(factors))],
        })
        st.subheader("Selection Frequency")
        st.dataframe(count_df, use_container_width=True, hide_index=True)

        tab1, tab2, tab3, tab4 = st.tabs([
            "Initial f_B→other",
            "Initial other→f_W",
            "Converted f_B→other",
            "Converted other→f_W",
        ])

        with tab1:
            st.dataframe(build_vector_table(transformed_experts, factors, "bto_init", expert_labels), use_container_width=True, hide_index=True)
        with tab2:
            st.dataframe(build_vector_table(transformed_experts, factors, "otw_init", expert_labels), use_container_width=True, hide_index=True)
        with tab3:
            st.dataframe(build_vector_table(transformed_experts, factors, "bto_trans", expert_labels), use_container_width=True, hide_index=True)
        with tab4:
            st.dataframe(build_vector_table(transformed_experts, factors, "otw_trans", expert_labels), use_container_width=True, hide_index=True)

        st.subheader("Aggregated Converted Vectors Used in Optimization")
        agg_df = build_aggregate_table(factors, agg_best, agg_worst)
        st.dataframe(agg_df, use_container_width=True, hide_index=True)
        st.caption("These TFNs are used directly in the optimization model. No defuzzification is applied before solving BWM.")

        result_df = pd.DataFrame({
            "Factor": factors,
            "Weight TFN": [tfn_to_str(w, 6) for w in weights],
            "GMI": [round(gmi(w), 6) for w in weights],
        }).sort_values("GMI", ascending=False)

        st.subheader("Final Fuzzy BWM Weights")
        st.dataframe(result_df, use_container_width=True, hide_index=True)
        st.bar_chart(result_df.set_index("Factor")[["GMI"]])

        st.session_state["bwm_weights"] = weights
        st.session_state["bwm_factors"] = factors
        st.session_state["bwm_common_best"] = common_best_idx
        st.session_state["bwm_common_worst"] = common_worst_idx
        st.session_state["bwm_agg_best"] = agg_best
        st.session_state["bwm_agg_worst"] = agg_worst

# ============================================================
# MODULE 3: FUZZY LBWA + HYBRID
# ============================================================
elif module == "3) Fuzzy LBWA + Hybrid":
    st.markdown('<div class="section-head">Fuzzy LBWA and Hybrid Weighting</div>', unsafe_allow_html=True)

    use_bwm_factors = st.toggle("Use factors from BWM", value=True)

    if use_bwm_factors and "bwm_factors" in st.session_state:
        factors = st.session_state["bwm_factors"]
        reference_idx_default = st.session_state.get("bwm_common_best", 0)
    else:
        n_f = st.number_input("Number of factors", min_value=2, value=5, step=1)
        factor_text = st.text_area(
            "Factor names (one per line)",
            value="\n".join([f"F{i+1}" for i in range(n_f)]),
            height=120,
        )
        factors = parse_names(factor_text, n_f, "F")
        reference_idx_default = 0

    n_exp_lbwa = st.number_input("Number of experts", min_value=1, value=4, step=1)
    theta = st.number_input("Theta (θ)", min_value=0.0001, value=2.1, step=0.1, format="%.4f")
    use_sample_lbwa = st.toggle("Use sample LBWA table", value=False)

    st.markdown("**Step A: Enter Factor Information**")
    st.caption("Edit one table directly for factor names, Qi values, and expert scores, following the FLBWA app style.")

    lbwa_sig = ("|".join(factors), n_exp_lbwa, use_sample_lbwa)
    if st.button("Prepare / Refresh LBWA Table", key="prep_lbwa") or st.session_state.get("lbwa_sig") != lbwa_sig:
        st.session_state["lbwa_sig"] = lbwa_sig
        st.session_state["lbwa_editor_df"] = init_lbwa_editor_df(factors, n_exp_lbwa, use_sample_lbwa, seed=200)

    base_df = st.session_state["lbwa_editor_df"].copy()

    # keep table shape aligned with current factor/expert counts
    if list(base_df["Factor"]) != factors or len([c for c in base_df.columns if c.startswith("E")]) != n_exp_lbwa:
        base_df = init_lbwa_editor_df(factors, n_exp_lbwa, use_sample_lbwa, seed=200)
        st.session_state["lbwa_editor_df"] = base_df

    edited_lbwa_df = st.data_editor(
        base_df,
        key="lbwa_editor",
        use_container_width=True,
        num_rows="fixed",
        hide_index=True,
        column_config={
            "Factor": st.column_config.TextColumn("Factor", required=True),
            "Qi": st.column_config.NumberColumn("Qi", min_value=0.0, step=1.0, format="%.2f"),
            **{
                f"E{i+1}": st.column_config.NumberColumn(
                    f"E{i+1}", min_value=0.0, step=0.1, format="%.4f"
                )
                for i in range(n_exp_lbwa)
            }
        }
    )

    st.markdown("**Step B: Select Reference / Main factor**")
    factor_options = list(range(len(edited_lbwa_df)))

    def factor_label_func(idx):
        val = str(edited_lbwa_df.iloc[idx]["Factor"]).strip()
        return f"{idx+1}. {val if val else f'Factor {idx+1}'}"

    reference_idx_lbwa = st.selectbox(
        "Reference / Main factor for LBWA",
        options=factor_options,
        index=min(reference_idx_default, len(factor_options) - 1),
        format_func=factor_label_func,
    )

    render_scale_table(HYBRID_SCALE, HYBRID_MEANING, "Show hybrid priority code legend")

    c1, c2 = st.columns(2)
    with c1:
        alpha_code = st.selectbox("Priority of FBWM weight (α)", options=list(HYBRID_SCALE.keys()), index=3)
    with c2:
        beta_code = st.selectbox("Priority of FLBWA weight (β)", options=list(HYBRID_SCALE.keys()), index=3)

    alpha_tfn = HYBRID_SCALE[alpha_code]
    beta_tfn = HYBRID_SCALE[beta_code]

    if st.button("Compute LBWA and Hybrid Weights", type="primary"):
        try:
            lbwa_output = run_lbwa_excel_single_table(
                input_df=edited_lbwa_df,
                num_experts=n_exp_lbwa,
                theta=float(theta),
                reference_idx=reference_idx_lbwa,
            )

            lbwa_weights = lbwa_output["weights"]
            result_df = lbwa_output["result_df"]
            top_factor = lbwa_output["top_factor"]
            top_weight = lbwa_output["top_weight"]
            crisp_sum = lbwa_output["crisp_sum"]

            st.subheader("Fuzzy LBWA Weights")

            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Reference Factor", lbwa_output["input_df"].iloc[reference_idx_lbwa]["Factor"])
            with c2:
                st.metric("Sum of Crisp Values", f"{crisp_sum:.10f}")
            with c3:
                st.metric("Sum of Final Weights", f"{result_df['Normalized Weight'].sum():.10f}")

            st.markdown(
                f'<div class="top-factor">🏆 Highest-ranked factor: {top_factor} '
                f'&nbsp;&nbsp;|&nbsp;&nbsp; Weight = {top_weight:.6f}</div>',
                unsafe_allow_html=True
            )

            tab1, tab2, tab3, tab4 = st.tabs([
                "TFN",
                "Influence",
                "Fuzzy Weights",
                "Final Results"
            ])

            with tab1:
                st.dataframe(lbwa_output["tfn_df"], use_container_width=True, hide_index=True)

            with tab2:
                st.dataframe(lbwa_output["influence_df"], use_container_width=True, hide_index=True)

            with tab3:
                st.dataframe(lbwa_output["fuzzy_weight_df"], use_container_width=True, hide_index=True)

            with tab4:
                styled_result = (
                    result_df.style
                    .apply(highlight_top_factor, axis=1)
                    .format({
                        "Qi": "{:.2f}",
                        "Crisp Value": "{:.10f}",
                        "Normalized Weight": "{:.10f}",
                    })
                )
                st.dataframe(styled_result, use_container_width=True, hide_index=True)

            st.bar_chart(result_df.set_index("Factor")[["Normalized Weight"]])

            st.session_state["lbwa_weights"] = lbwa_weights
            st.session_state["lbwa_factors"] = lbwa_output["input_df"]["Factor"].tolist()
            st.session_state["lbwa_result_df"] = result_df

            if "bwm_weights" in st.session_state and len(st.session_state["bwm_weights"]) == len(lbwa_weights):
                hybrid_weights = combine_weights(
                    st.session_state["bwm_weights"],
                    lbwa_weights,
                    alpha_tfn,
                    beta_tfn
                )

                hybrid_df = pd.DataFrame({
                    "Factor": lbwa_output["input_df"]["Factor"].tolist(),
                    "Hybrid TFN": [tfn_to_str(w, 6) for w in hybrid_weights],
                    "Hybrid GMI": [round(gmi(w), 6) for w in hybrid_weights],
                }).sort_values("Hybrid GMI", ascending=False)

                st.subheader("Hybrid Weights (FBWM + FLBWA)")
                st.dataframe(hybrid_df, use_container_width=True, hide_index=True)
                st.bar_chart(hybrid_df.set_index("Factor")[["Hybrid GMI"]])

                st.session_state["hybrid_weights"] = hybrid_weights
            else:
                st.warning("No compatible BWM weights found. Hybrid result is not computed.")
                st.session_state["hybrid_weights"] = lbwa_weights

        except Exception as e:
            st.error(f"An error occurred while computing LBWA: {e}")

# ============================================================
# MODULE 4: FUZZY BONFERRONI COCO-SO
# ============================================================
elif module == "4) Fuzzy Bonferroni CoCoSo":
    st.markdown('<div class="section-head">Fuzzy Bonferroni CoCoSo – Technology Ranking</div>', unsafe_allow_html=True)

    st.info("Bonferroni CoCoSo uses crisp criterion weights after GMI defuzzification of fuzzy weights.")

    use_sample_alt = st.toggle("Use sample alternatives", value=True)
    alt_names = (
        ["TS-SS", "TS-HP", "TS-PCC", "TS-MPP"]
        if use_sample_alt
        else [x.strip() for x in st.text_area("Alternative names (one per line)", value="A1\nA2\nA3", height=120).splitlines() if x.strip()]
    )

    if len(alt_names) == 0:
        st.warning("Please define at least one alternative.")
        st.stop()

    use_existing_weights = "hybrid_weights" in st.session_state
    if use_existing_weights:
        criteria_names = st.session_state.get(
            "bwm_factors",
            [f"C{i+1}" for i in range(len(st.session_state["hybrid_weights"]))],
        )
        init_weights = st.session_state["hybrid_weights"]
    else:
        n_crit_manual = st.number_input("Number of criteria", min_value=2, value=5, step=1)
        crit_text = st.text_area(
            "Criterion names (one per line)",
            value="\n".join([f"C{i+1}" for i in range(n_crit_manual)]),
            height=120,
        )
        criteria_names = parse_names(crit_text, n_crit_manual, "C")
        init_weights = None

    cocoso_sig = ("|".join(criteria_names), "|".join(alt_names), use_existing_weights)
    if st.button("Prepare / Refresh Bonferroni CoCoSo Tables", key="prep_cocoso") or st.session_state.get("cocoso_sig") != cocoso_sig:
        st.session_state["cocoso_sig"] = cocoso_sig
        st.session_state["cocoso_criteria_df"] = init_criteria_weight_df(criteria_names, init_weights)
        for a, alt in enumerate(alt_names):
            st.session_state[f"cocoso_alt_df_{a}"] = init_decision_df(criteria_names, use_sample=False, seed=300 + a)

    criteria_df = st.session_state["cocoso_criteria_df"].copy()
    criteria_df["Criterion"] = criteria_names

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
            },
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
            },
        )

    preview_df = pd.DataFrame({
        "Criterion": edited_criteria["Criterion"].tolist(),
        "Crisp weight (GMI)": [defuzz_tfn((float(r["w_l"]), float(r["w_m"]), float(r["w_u"]))) for _, r in edited_criteria.iterrows()],
    })
    st.dataframe(preview_df, use_container_width=True, hide_index=True)

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
                },
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
            fuzzy_df, types, crisp_weights, float(phi1), float(phi2), float(pi)
        )

        st.subheader("Final Ranking")
        st.dataframe(ranking_df[["Rank", "Alternative", "Final_l", "Final_m", "Final_u", "Crisp"]], use_container_width=True, hide_index=True)
        st.bar_chart(ranking_df.set_index("Alternative")[["Crisp"]])

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
