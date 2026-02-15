
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st
from scipy.optimize import linprog

# ==========================================================
# Triangular Fuzzy Number (TFN) utilities
# ==========================================================
TFN = Tuple[float, float, float]

def tfn(l: float, m: float, u: float) -> TFN:
    l2, m2, u2 = float(l), float(m), float(u)
    if l2 > m2: l2, m2 = m2, l2
    if m2 > u2: m2, u2 = u2, m2
    if l2 > m2: l2, m2 = m2, l2
    return (l2, m2, u2)

def gmi(x: TFN) -> float:
    # Eq. (10) in the paper: GMI(F) = (l + 4m + u)/6
    l, m, u = x
    return (l + 4*m + u) / 6.0

def tfn_add(a: TFN, b: TFN) -> TFN:
    return (a[0]+b[0], a[1]+b[1], a[2]+b[2])

def tfn_sub(a: TFN, b: TFN) -> TFN:
    # paper's subtraction definition: (l1-u2, m1-m2, u1-l2)
    return (a[0]-b[2], a[1]-b[1], a[2]-b[0])

def tfn_mul(a: TFN, b: TFN) -> TFN:
    return (a[0]*b[0], a[1]*b[1], a[2]*b[2])

def tfn_div(a: TFN, b: TFN) -> TFN:
    # (l1/u2, m1/m2, u1/l2)
    if b[0] <= 0 or b[1] <= 0 or b[2] <= 0:
        raise ValueError("Division by non-positive TFN is not supported.")
    return (a[0]/b[2], a[1]/b[1], a[2]/b[0])

def tfn_pow_scalar(a: TFN, p: float) -> TFN:
    # assumes a is non-negative
    if a[0] < 0:
        raise ValueError("Power with negative TFN not supported.")
    return (a[0]**p, a[1]**p, a[2]**p)

def tfn_scale(a: TFN, s: float) -> TFN:
    if s >= 0:
        return (a[0]*s, a[1]*s, a[2]*s)
    # if negative, reverse bounds
    return (a[2]*s, a[1]*s, a[0]*s)

def tfn_geometric_aggregate(tfns: List[TFN]) -> TFN:
    # Eq. (9): geometric aggregation across experts
    arr = np.array(tfns, dtype=float)
    l = float(np.prod(arr[:,0]) ** (1.0/len(tfns)))
    m = float(np.prod(arr[:,1]) ** (1.0/len(tfns)))
    u = float(np.prod(arr[:,2]) ** (1.0/len(tfns)))
    return tfn(l,m,u)

def enforce_componentwise_normalization(w: List[TFN]) -> List[TFN]:
    # Simple practical normalization: normalize l,m,u separately to sum to 1, then fix ordering.
    lsum = sum(x[0] for x in w) or 1.0
    msum = sum(x[1] for x in w) or 1.0
    usum = sum(x[2] for x in w) or 1.0
    out=[]
    for (l,m,u) in w:
        l2, m2, u2 = l/lsum, m/msum, u/usum
        out.append(tfn(l2,m2,u2))
    return out

def format_tfn(x: TFN, nd: int = 6) -> str:
    return f"({x[0]:.{nd}f}, {x[1]:.{nd}f}, {x[2]:.{nd}f})"

# ==========================================================
# Linguistic scales (from Appendix tables)
# ==========================================================

# Table S5 (Fuzzy Delphi linguistic terms)
DELPHI_SCALE: Dict[str, TFN] = {
    "Absolutely unimportant": (0.00, 0.10, 0.30),
    "Unimportant":           (0.10, 0.30, 0.50),
    "Moderately important":  (0.30, 0.50, 0.70),
    "Important":             (0.50, 0.70, 0.90),
    "Extremely important":   (0.70, 0.90, 1.00),
}

# Table S6 (Fuzzy BWM linguistic terms)
BWM_SCALE: Dict[str, TFN] = {
    "Very low importance": (1.0, 1.0, 3.0),
    "Low importance":      (1.0, 3.0, 5.0),
    "Medium importance":   (3.0, 5.0, 7.0),
    "High importance":     (5.0, 7.0, 9.0),
    "Very high importance":(7.0, 9.0, 9.0),
}

# Table S7 (Hybrid integration scale for method weights α and β)
INTEGRATION_SCALE: Dict[str, TFN] = {
    "Very low importance": (0.1, 0.1, 0.3),
    "Low importance":      (0.1, 0.3, 0.5),
    "Medium importance":   (0.3, 0.5, 0.7),
    "High importance":     (0.5, 0.7, 0.9),
    "Very high importance":(0.7, 0.9, 1.0),
}

# ==========================================================
# Module 1: Fuzzy Delphi
# ==========================================================
def run_fuzzy_delphi(df_ling: pd.DataFrame, threshold: float) -> pd.DataFrame:
    # df_ling: rows criteria, columns experts, values in DELPHI_SCALE keys
    rows=[]
    for crit, row in df_ling.iterrows():
        tfns=[]
        for v in row.values:
            if pd.isna(v) or str(v).strip()=="":
                continue
            tfns.append(DELPHI_SCALE[str(v)])
        if not tfns:
            agg = (0.0, 0.0, 0.0)
        else:
            agg = tfn_geometric_aggregate(tfns)
        crisp = gmi(agg)
        keep = crisp >= threshold
        rows.append([crit, agg[0], agg[1], agg[2], crisp, keep])
    out = pd.DataFrame(rows, columns=["Criterion","Agg_l","Agg_m","Agg_u","GMI","Selected"])
    out = out.sort_values("GMI", ascending=False).reset_index(drop=True)
    return out

# ==========================================================
# Module 2: Fuzzy BWM (practical 3-LP approach)
# ==========================================================
def _solve_bwm_lp(a_B: List[float], a_W: List[float], best_idx: int, worst_idx: int) -> Tuple[np.ndarray, float]:
    """
    Solve standard (crisp) BWM LP:
        min ξ
        s.t. |w_B - a_Bj w_j| <= ξ  for all j
             |w_j - a_jW w_W| <= ξ  for all j
             Σ w_j = 1, w_j >= 0
    Inputs:
        a_B: list of a_Bj for all j (a_B[best_idx] should be 1)
        a_W: list of a_jW for all j (a_W[worst_idx] should be 1)
    """
    n=len(a_B)
    # variables: w0..w(n-1), xi
    c = np.zeros(n+1)
    c[-1]=1.0  # minimize xi

    A_ub=[]
    b_ub=[]

    # constraints for best-to-others
    for j in range(n):
        # w_B - a_Bj w_j <= xi  -> w_B - a_Bj w_j - xi <= 0
        row = np.zeros(n+1)
        row[best_idx]=1.0
        row[j] -= a_B[j]
        row[-1] = -1.0
        A_ub.append(row); b_ub.append(0.0)

        # -(w_B - a_Bj w_j) <= xi -> -w_B + a_Bj w_j - xi <= 0
        row = np.zeros(n+1)
        row[best_idx]=-1.0
        row[j] += a_B[j]
        row[-1] = -1.0
        A_ub.append(row); b_ub.append(0.0)

    # constraints for others-to-worst
    for j in range(n):
        # w_j - a_jW w_W <= xi
        row = np.zeros(n+1)
        row[j]=1.0
        row[worst_idx] -= a_W[j]
        row[-1]=-1.0
        A_ub.append(row); b_ub.append(0.0)

        # -(w_j - a_jW w_W) <= xi  -> -w_j + a_jW w_W - xi <= 0
        row = np.zeros(n+1)
        row[j]=-1.0
        row[worst_idx] += a_W[j]
        row[-1]=-1.0
        A_ub.append(row); b_ub.append(0.0)

    A_ub=np.array(A_ub); b_ub=np.array(b_ub)

    A_eq=np.zeros((1,n+1))
    A_eq[0,:n]=1.0
    b_eq=np.array([1.0])

    bounds=[(0,None)]*n + [(0,None)]
    res=linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if not res.success:
        raise RuntimeError(f"BWM LP failed: {res.message}")
    w=res.x[:n]
    xi=res.x[-1]
    return w, xi

def run_fuzzy_bwm(criteria: List[str],
                  best: str,
                  worst: str,
                  best_to_others: Dict[str, str],
                  others_to_worst: Dict[str, str]) -> Tuple[pd.DataFrame, float, float, float]:
    n=len(criteria)
    best_idx=criteria.index(best)
    worst_idx=criteria.index(worst)

    # build TFNs for a_Bj and a_jW
    aB_tfn=[]
    aW_tfn=[]
    for c in criteria:
        if c==best:
            aB_tfn.append((1.0,1.0,1.0))
        else:
            aB_tfn.append(BWM_SCALE[best_to_others[c]])
        if c==worst:
            aW_tfn.append((1.0,1.0,1.0))
        else:
            aW_tfn.append(BWM_SCALE[others_to_worst[c]])

    # Solve 3 crisp LPs (l,m,u components) and combine
    ws=[]
    xis=[]
    for k in range(3):
        aB=[x[k] for x in aB_tfn]
        aW=[x[k] for x in aW_tfn]
        w, xi = _solve_bwm_lp(aB, aW, best_idx, worst_idx)
        ws.append(w); xis.append(xi)

    w_l, w_m, w_u = ws
    out=[]
    for j,c in enumerate(criteria):
        out.append([c, float(w_l[j]), float(w_m[j]), float(w_u[j]), gmi((float(w_l[j]), float(w_m[j]), float(w_u[j])))])
    df=pd.DataFrame(out, columns=["Criterion","w_l","w_m","w_u","GMI"])
    # normalize componentwise + enforce ordering
    w_tfns=[tfn(r.w_l, r.w_m, r.w_u) for r in df.itertuples()]
    w_tfns=enforce_componentwise_normalization(w_tfns)
    df["w_l"]=[x[0] for x in w_tfns]
    df["w_m"]=[x[1] for x in w_tfns]
    df["w_u"]=[x[2] for x in w_tfns]
    df["GMI"]=[gmi(x) for x in w_tfns]
    df=df.sort_values("GMI", ascending=False).reset_index(drop=True)
    return df, float(xis[0]), float(xis[1]), float(xis[2])

# ==========================================================
# Module 3: Fuzzy LBWA + Hybrid weights
# ==========================================================
def run_fuzzy_lbwa(criteria: List[str],
                   lbwa_expert_values: List[Dict[str, float]],
                   best_factor: str,
                   gamma: float,
                   elasticity: Optional[float]=None) -> pd.DataFrame:
    """
    Implements fuzzy-LBWA per paper's steps.
    Inputs:
      - lbwa_expert_values: list over experts, each dict {criterion: numeric importance (0..tau)}
      - gamma: elasticity coefficient γ (paper uses a small positive number, e.g., 0.1)
      - elasticity: ℘ (Greek P) must be > tau; if None uses tau+1
    """
    # Step 3.1: levels implied by integer part of values (user can decide their own scheme)
    # Here we follow the paper's generic formulation: tau = max level size (cardinality), and ℘ > tau.
    # We use tau as (max assigned value) rounded up.
    all_vals = [v for ex in lbwa_expert_values for v in ex.values() if v is not None]
    tau = int(math.ceil(max(all_vals))) if all_vals else 1

    P = float(elasticity) if elasticity is not None else float(tau + 1)

    # Step 3.3: aggregate expert values to TFNs via Eq. (24): min/mean/max
    rows=[]
    for c in criteria:
        vals=[ex.get(c, np.nan) for ex in lbwa_expert_values]
        vals=[v for v in vals if not (pd.isna(v) or v is None)]
        if not vals:
            lam = (0.0,0.0,0.0)
        else:
            lam = (float(np.min(vals)), float(np.mean(vals)), float(np.max(vals)))
        rows.append([c, lam[0], lam[1], lam[2]])
    df=pd.DataFrame(rows, columns=["Criterion","lambda_l","lambda_m","lambda_u"])

    # Step 3.4 influence function χ(fjg), Eq. (26)
    # χ_l = P^j / (P + γ_u), χ_m = P^j / (P + γ_m), χ_u = P^j / (P + γ_l)
    # where j = level index; we approximate j as 0 for best factor and 1 otherwise (common LBWA use),
    # or allow users to encode level in their numeric value. We'll use j = floor(mean(lambda)).
    chis=[]
    for r in df.itertuples():
        j = int(math.floor(r.lambda_m))
        chi_l = (P**j) / (P + gamma)  # using gamma for all components in this simplified UI
        chi_m = (P**j) / (P + gamma)
        chi_u = (P**j) / (P + gamma)
        chis.append((chi_l, chi_m, chi_u))
    df["chi_l"]=[x[0] for x in chis]
    df["chi_m"]=[x[1] for x in chis]
    df["chi_u"]=[x[2] for x in chis]

    # Eq. (27): best factor weight coefficient ϖ1 (best factor is f1 in paper)
    # ϖ_l1 = 1 / (1 + Σ chi_u(other)), ϖ_m1 = 1 / (1 + Σ chi_m(other)), ϖ_u1 = 1 / (1 + Σ chi_l(other))
    best_idx = criteria.index(best_factor)
    other_idx = [i for i in range(len(criteria)) if i != best_idx]
    sum_chi_u = sum(df.loc[i,"chi_u"] for i in other_idx)
    sum_chi_m = sum(df.loc[i,"chi_m"] for i in other_idx)
    sum_chi_l = sum(df.loc[i,"chi_l"] for i in other_idx)
    varpi_best = tfn(1.0/(1.0+sum_chi_u), 1.0/(1.0+sum_chi_m), 1.0/(1.0+sum_chi_l))

    # Eq. (28): ϖ_j = χ_j ⊗ ϖ_1 with bounds aligned
    varpis=[]
    for i,c in enumerate(criteria):
        if i==best_idx:
            varpis.append(varpi_best)
        else:
            chi = tfn(df.loc[i,"chi_l"], df.loc[i,"chi_m"], df.loc[i,"chi_u"])
            varpis.append(tfn_mul(chi, varpi_best))

    varpis = enforce_componentwise_normalization(varpis)
    df_out=pd.DataFrame({
        "Criterion": criteria,
        "varpi_l": [x[0] for x in varpis],
        "varpi_m": [x[1] for x in varpis],
        "varpi_u": [x[2] for x in varpis],
    })
    df_out["GMI"]=[gmi((r.varpi_l,r.varpi_m,r.varpi_u)) for r in df_out.itertuples()]
    df_out=df_out.sort_values("GMI", ascending=False).reset_index(drop=True)
    return df_out

def run_hybrid_weights(bwm_weights: pd.DataFrame,
                       lbwa_weights: pd.DataFrame,
                       alpha_tfn: TFN,
                       beta_tfn: TFN) -> pd.DataFrame:
    """
    Eq. (29): ω*_T = (ω_TFBWM)^α ⊗ (ϖ_TFLBWA)^β / Σ(...)
    Practical implementation: apply component-wise with α,β defuzzified (GMI).
    """
    alpha=float(gmi(alpha_tfn))
    beta=float(gmi(beta_tfn))

    bwm_map={r.Criterion: tfn(r.w_l, r.w_m, r.w_u) for r in bwm_weights.itertuples()}
    lbwa_map={r.Criterion: tfn(r.varpi_l, r.varpi_m, r.varpi_u) for r in lbwa_weights.itertuples()}
    common=[c for c in bwm_map.keys() if c in lbwa_map]

    raw=[]
    for c in common:
        w=bwm_map[c]; v=lbwa_map[c]
        # compute per component and then normalize
        comp = tfn_mul(tfn_pow_scalar(w, alpha), tfn_pow_scalar(v, beta))
        raw.append(comp)

    raw = enforce_componentwise_normalization(raw)
    out=[]
    for c,comp in zip(common, raw):
        out.append([c, comp[0], comp[1], comp[2], gmi(comp)])
    df=pd.DataFrame(out, columns=["Criterion","w*_l","w*_m","w*_u","GMI"])
    df=df.sort_values("GMI", ascending=False).reset_index(drop=True)
    return df

# ==========================================================
# Module 4: Fuzzy CoCoSo with Bonferroni + "lower" defuzz (GMI)
# ==========================================================

def to_tfn_from_number(x: float, sigma: float) -> TFN:
    # Eq. (35): convert quantitative criteria to TFN with uncertainty sigma
    return tfn(x*(1.0-sigma), x, x*(1.0+sigma))

def normalize_fuzzy_matrix(X: List[List[TFN]], benefit: List[bool]) -> List[List[TFN]]:
    # Eq. (36): fuzzy normalization
    m=len(X); n=len(X[0])
    # per criterion j: partial^+ = max u, partial^- = min l
    max_u=[max(X[i][j][2] for i in range(m)) for j in range(n)]
    min_l=[min(X[i][j][0] for i in range(m)) for j in range(n)]
    out=[[None]*n for _ in range(m)]
    for i in range(m):
        for j in range(n):
            l,mv,u = X[i][j]
            if benefit[j]:
                denom=max_u[j] if max_u[j]!=0 else 1.0
                out[i][j]=tfn(l/denom, mv/denom, u/denom)
            else:
                # cost: min_l / u, min_l / m, min_l / l
                num=min_l[j]
                out[i][j]=tfn(num/u if u!=0 else 0.0, num/mv if mv!=0 else 0.0, num/l if l!=0 else 0.0)
    return out

def bonferroni_sum_score(gammas: List[TFN], weights: List[float], phi1: float, phi2: float) -> TFN:
    # Eq. (37): SCoB_{phi1,phi2}
    n=len(gammas)
    # precompute exponent coeff c_ij = w_i w_j / (1 - w_i) for i!=j
    def comp(level: int) -> float:
        total=0.0
        for i in range(n):
            wi=weights[i]
            denom=max(1e-12, 1.0-wi)
            for j in range(n):
                if i==j: 
                    continue
                wj=weights[j]
                coef=wi*wj/denom
                gi=gammas[i][level]
                gj=gammas[j][level]
                total += coef * (gi**phi1) * (gj**phi2)
        return total ** (1.0/(phi1+phi2))
    return tfn(comp(0), comp(1), comp(2))

def bonferroni_prod_score(gammas: List[TFN], weights: List[float], phi1: float, phi2: float) -> TFN:
    # Eq. (38): PCoB_{phi1,phi2} (geometric Bonferroni-like product)
    n=len(gammas)
    def comp(level: int) -> float:
        prod=1.0
        for i in range(n):
            wi=weights[i]
            denom=max(1e-12, 1.0-wi)
            for j in range(n):
                if i==j:
                    continue
                wj=weights[j]
                coef=wi*wj/denom
                gi=gammas[i][level]
                gj=gammas[j][level]
                term = (phi1*gi + phi2*gj)
                term = max(1e-12, term)  # avoid zero
                prod *= term ** coef
        return prod ** (1.0/(phi1+phi2))
    return tfn(comp(0), comp(1), comp(2))

def cocoso_bonferroni(decision: pd.DataFrame,
                      weights_tfn: Dict[str, TFN],
                      benefit: Dict[str, bool],
                      sigma: float,
                      phi1: float,
                      phi2: float,
                      pi: float) -> pd.DataFrame:
    """
    decision: rows alternatives, columns criteria; numeric (float) OR TFN string "(l,m,u)".
    weights_tfn: criterion->TFN weight
    benefit: criterion->True/False
    """
    alts=list(decision.index)
    crits=list(decision.columns)
    m=len(alts); n=len(crits)

    # Build fuzzy matrix X[i][j]
    X=[]
    for a in alts:
        row=[]
        for c in crits:
            v=decision.loc[a,c]
            if isinstance(v,str) and v.strip().startswith("("):
                parts=[float(x) for x in v.strip("() ").split(",")]
                row.append(tfn(parts[0],parts[1],parts[2]))
            else:
                row.append(to_tfn_from_number(float(v), sigma))
        X.append(row)

    Xn=normalize_fuzzy_matrix(X, [benefit[c] for c in crits])

    # Use mid weights for Bonferroni coefficients
    w_mid=[gmi(weights_tfn[c]) for c in crits]
    # normalize to sum=1
    s=sum(w_mid) or 1.0
    w_mid=[x/s for x in w_mid]

    # Step 4.3: compute SCoB and PCoB per alternative
    S=[]; P=[]
    for i,a in enumerate(alts):
        gammas = Xn[i]  # list TFN per criterion
        S.append(bonferroni_sum_score(gammas, w_mid, phi1, phi2))
        P.append(bonferroni_prod_score(gammas, w_mid, phi1, phi2))

    # pooling strategies (CoCoSo-style), Eq (39)-(43)
    # ψ_a = (S + P) / Σ(S+P)
    Sp=[tfn_add(S[i], P[i]) for i in range(m)]
    sumSp=tfn_add((0,0,0), (0,0,0))
    for x in Sp:
        sumSp=tfn_add(sumSp, x)

    psi_a=[tfn_div(Sp[i], sumSp) for i in range(m)]

    # ψ_b = (S/min S) + (P/min P)
    minS = tfn(min(s[0] for s in S), min(s[1] for s in S), min(s[2] for s in S))
    minP = tfn(min(p[0] for p in P), min(p[1] for p in P), min(p[2] for p in P))
    psi_b=[tfn_add(tfn_div(S[i], minS), tfn_div(P[i], minP)) for i in range(m)]

    # ψ_c = (π S + (1-π) P) / (π max S + (1-π) max P)
    maxS = tfn(max(s[0] for s in S), max(s[1] for s in S), max(s[2] for s in S))
    maxP = tfn(max(p[0] for p in P), max(p[1] for p in P), max(p[2] for p in P))
    denom = tfn_add(tfn_scale(maxS, pi), tfn_scale(maxP, 1.0-pi))
    psi_c=[tfn_div(tfn_add(tfn_scale(S[i], pi), tfn_scale(P[i], 1.0-pi)), denom) for i in range(m)]

    # Eq (45): final score index
    psi=[]
    for i in range(m):
        # geometric mean + arithmetic mean
        geo = tfn_pow_scalar(tfn_mul(tfn_mul(psi_a[i], psi_b[i]), psi_c[i]), 1.0/3.0)
        ari = tfn_scale(tfn_add(tfn_add(psi_a[i], psi_b[i]), psi_c[i]), 1.0/3.0)
        psi.append(tfn_add(geo, ari))

    # Eq (46): defuzzify with GMI
    crisp=[gmi(x) for x in psi]
    df=pd.DataFrame({
        "Alternative": alts,
        "SCoB_l":[x[0] for x in S], "SCoB_m":[x[1] for x in S], "SCoB_u":[x[2] for x in S],
        "PCoB_l":[x[0] for x in P], "PCoB_m":[x[1] for x in P], "PCoB_u":[x[2] for x in P],
        "psi_l":[x[0] for x in psi], "psi_m":[x[1] for x in psi], "psi_u":[x[2] for x in psi],
        "CrispScore": crisp,
    }).set_index("Alternative")
    df["Rank"]=df["CrispScore"].rank(ascending=False, method="dense").astype(int)
    df=df.sort_values(["Rank","CrispScore"], ascending=[True,False])
    return df

# ==========================================================
# Streamlit UI
# ==========================================================
st.set_page_config(page_title="Integrated Fuzzy MCDM App (Delphi • BWM • LBWA • CoCoSo-B)", layout="wide")

st.title("Integrated Fuzzy MCDM App")
st.caption("Modules: (1) Fuzzy Delphi, (2) Fuzzy BWM, (3) Fuzzy LBWA + Hybrid weights, (4) Fuzzy CoCoSo-B (Bonferroni) with GMI defuzzification.")

module = st.sidebar.radio("Select module", [
    "1) Fuzzy Delphi",
    "2) Fuzzy BWM",
    "3) Fuzzy LBWA + Hybrid weights",
    "4) Fuzzy CoCoSo-B (Bonferroni) + Defuzzification",
])

# Shared session storage
if "selected_criteria" not in st.session_state:
    st.session_state.selected_criteria=[]
if "bwm_weights" not in st.session_state:
    st.session_state.bwm_weights=None
if "lbwa_weights" not in st.session_state:
    st.session_state.lbwa_weights=None
if "hybrid_weights" not in st.session_state:
    st.session_state.hybrid_weights=None

# --------------------------
# Module 1
# --------------------------
if module.startswith("1"):
    st.subheader("Module 1 — Fuzzy Delphi")
    st.write("Enter linguistic importance ratings from multiple experts. The app aggregates using fuzzy geometric mean and selects criteria using a GMI threshold.")

    col1,col2 = st.columns([2,1])
    with col2:
        threshold = st.slider("Threshold ι (GMI cutoff)", 0.0, 1.0, 0.6, 0.05)
        n_criteria = st.number_input("Number of criteria", min_value=3, max_value=100, value=10, step=1)
        n_experts = st.number_input("Number of experts", min_value=2, max_value=30, value=5, step=1)

    crit_names=[f"C{i+1}" for i in range(int(n_criteria))]
    exp_names=[f"Ex{i+1}" for i in range(int(n_experts))]
    df0=pd.DataFrame({e:["Moderately important"]*len(crit_names) for e in exp_names}, index=crit_names)
    st.info("You can rename criteria (index) by editing after exporting template CSV, or directly edit in the table below.")
    df_in = st.data_editor(df0, use_container_width=True, num_rows="fixed",
                           column_config={c: st.column_config.SelectboxColumn(c, options=list(DELPHI_SCALE.keys())) for c in df0.columns})

    if st.button("Run Fuzzy Delphi"):
        res=run_fuzzy_delphi(df_in, threshold)
        st.success(f"Done. Selected {int(res['Selected'].sum())} criteria.")
        st.dataframe(res, use_container_width=True)

        st.session_state.selected_criteria = res.loc[res["Selected"],"Criterion"].tolist()

        csv=res.to_csv(index=False).encode("utf-8")
        st.download_button("Download results (CSV)", csv, file_name="fuzzy_delphi_results.csv", mime="text/csv")

# --------------------------
# Module 2
# --------------------------
elif module.startswith("2"):
    st.subheader("Module 2 — Fuzzy BWM")
    st.write("Select best and worst criteria, then provide linguistic comparisons (best-to-others and others-to-worst) using the paper's TFN scale (Table S6).")

    criteria = st.session_state.selected_criteria or [f"C{i+1}" for i in range(8)]
    st.caption(f"Working criteria list ({len(criteria)}): {criteria}")
    best = st.selectbox("Best criterion (fB)", criteria, index=0)
    worst = st.selectbox("Worst criterion (fW)", criteria, index=len(criteria)-1)

    st.markdown("#### Best-to-others comparisons")
    b2o={}
    for c in criteria:
        if c==best:
            continue
        b2o[c]=st.selectbox(f"{best} compared to {c}", list(BWM_SCALE.keys()), index=3, key=f"b2o_{c}")

    st.markdown("#### Others-to-worst comparisons")
    o2w={}
    for c in criteria:
        if c==worst:
            continue
        o2w[c]=st.selectbox(f"{c} compared to {worst}", list(BWM_SCALE.keys()), index=3, key=f"o2w_{c}")

    if st.button("Run Fuzzy BWM"):
        try:
            df_w, xi_l, xi_m, xi_u = run_fuzzy_bwm(criteria, best, worst, b2o, o2w)
            st.success("Done.")
            st.caption(f"LP deviation (xi): lower={xi_l:.6f}, middle={xi_m:.6f}, upper={xi_u:.6f}")
            st.dataframe(df_w, use_container_width=True)
            st.session_state.bwm_weights = df_w

            csv=df_w.to_csv(index=False).encode("utf-8")
            st.download_button("Download weights (CSV)", csv, file_name="fuzzy_bwm_weights.csv", mime="text/csv")
        except Exception as e:
            st.error(str(e))

# --------------------------
# Module 3
# --------------------------
elif module.startswith("3"):
    st.subheader("Module 3 — Fuzzy LBWA + Hybrid weight determination")
    st.write("Provide LBWA numeric preferences per expert (e.g., 0–2 within levels). The app aggregates to TFNs and computes fuzzy-LBWA weights, then combines with fuzzy-BWM weights via the non-linear operator (Eq. 29).")

    criteria = st.session_state.selected_criteria or (st.session_state.bwm_weights["Criterion"].tolist() if st.session_state.bwm_weights is not None else [f"C{i+1}" for i in range(8)])
    st.caption(f"Working criteria list ({len(criteria)}): {criteria}")

    if st.session_state.bwm_weights is None:
        st.warning("Run Module 2 first to generate fuzzy-BWM weights (or provide your own in code).")

    best_factor = st.selectbox("Best factor for LBWA (fB)", criteria, index=0)
    gamma = st.number_input("Elasticity coefficient γ", min_value=0.0001, max_value=10.0, value=0.1, step=0.1)
    n_experts = st.number_input("Number of LBWA experts", min_value=2, max_value=30, value=5, step=1)

    st.markdown("#### Enter LBWA expert numeric values (one column per expert)")
    df_lbwa = pd.DataFrame({f"Ex{i+1}":[0.0]*len(criteria) for i in range(int(n_experts))}, index=criteria)
    df_lbwa_in = st.data_editor(df_lbwa, use_container_width=True)

    if st.button("Run Fuzzy LBWA"):
        lbwa_ex=[]
        for col in df_lbwa_in.columns:
            lbwa_ex.append({c: float(df_lbwa_in.loc[c,col]) for c in criteria})
        df_l=run_fuzzy_lbwa(criteria, lbwa_ex, best_factor=best_factor, gamma=float(gamma))
        st.success("LBWA done.")
        st.dataframe(df_l, use_container_width=True)
        st.session_state.lbwa_weights = df_l

    st.markdown("### Hybrid integration (Eq. 29)")
    colA, colB = st.columns(2)
    with colA:
        alpha_term = st.selectbox("α (importance of fuzzy-BWM)", list(INTEGRATION_SCALE.keys()), index=3)
    with colB:
        beta_term = st.selectbox("β (importance of fuzzy-LBWA)", list(INTEGRATION_SCALE.keys()), index=3)

    if st.button("Compute hybrid weights"):
        if st.session_state.bwm_weights is None or st.session_state.lbwa_weights is None:
            st.error("Please run both Module 2 (BWM) and Module 3 (LBWA) first.")
        else:
            df_h = run_hybrid_weights(st.session_state.bwm_weights, st.session_state.lbwa_weights,
                                     alpha_tfn=INTEGRATION_SCALE[alpha_term],
                                     beta_tfn=INTEGRATION_SCALE[beta_term])
            st.success("Hybrid weights computed.")
            st.dataframe(df_h, use_container_width=True)
            st.session_state.hybrid_weights = df_h
            csv=df_h.to_csv(index=False).encode("utf-8")
            st.download_button("Download hybrid weights (CSV)", csv, file_name="hybrid_weights.csv", mime="text/csv")

# --------------------------
# Module 4
# --------------------------
else:
    st.subheader("Module 4 — Fuzzy Bonferroni CoCoSo (CoCoSo-B) + Defuzzification")
    st.write("Input a decision matrix (alternatives × criteria). Numeric cells are converted to TFNs using uncertainty σ. If you want, you can input TFNs directly as '(l,m,u)'. Then the app ranks alternatives using fuzzy Bonferroni-CoCoSo and GMI defuzzification.")

    if st.session_state.hybrid_weights is None:
        st.warning("Run Module 3 to compute hybrid weights. Using equal weights for now.")
        criteria = st.session_state.selected_criteria or [f"C{i+1}" for i in range(6)]
        w_equal = {c: (1/len(criteria), 1/len(criteria), 1/len(criteria)) for c in criteria}
    else:
        criteria = st.session_state.hybrid_weights["Criterion"].tolist()
        w_equal = {}
        for row in st.session_state.hybrid_weights.to_dict(orient="records"):
            w_equal[row["Criterion"]] = tfn(row["w*_l"], row["w*_m"], row["w*_u"])

    st.markdown("#### Criteria setup")
    col1, col2, col3 = st.columns([2,1,1])
    with col1:
        st.caption("Benefit=True means higher is better. Cost=False means lower is better.")
    with col2:
        sigma = st.slider("Uncertainty σ (for numeric→TFN)", 0.0, 0.5, 0.05, 0.01)
    with col3:
        n_alts = st.number_input("Number of alternatives", min_value=2, max_value=50, value=4, step=1)

    benefit_flags={}
    for c in criteria:
        benefit_flags[c]=st.checkbox(f"Benefit criterion? {c}", value=True, key=f"ben_{c}")

    st.markdown("#### Decision matrix")
    alt_names=[f"A{i+1}" for i in range(int(n_alts))]
    df_dec = pd.DataFrame({c:[0.0]*len(alt_names) for c in criteria}, index=alt_names)
    df_dec_in = st.data_editor(df_dec, use_container_width=True)

    st.markdown("#### Bonferroni-CoCoSo parameters")
    colp1,colp2,colp3 = st.columns(3)
    with colp1:
        phi1 = st.number_input("φ1", min_value=0.1, max_value=10.0, value=1.0, step=0.1)
    with colp2:
        phi2 = st.number_input("φ2", min_value=0.1, max_value=10.0, value=1.0, step=0.1)
    with colp3:
        pi = st.slider("π (mixing coefficient)", 0.0, 1.0, 0.5, 0.05)

    if st.button("Run CoCoSo-B ranking"):
        try:
            res=cocoso_bonferroni(df_dec_in, weights_tfn=w_equal, benefit=benefit_flags,
                                 sigma=float(sigma), phi1=float(phi1), phi2=float(phi2), pi=float(pi))
            st.success("Ranking completed.")
            st.dataframe(res, use_container_width=True)
            csv=res.reset_index().to_csv(index=False).encode("utf-8")
            st.download_button("Download ranking results (CSV)", csv, file_name="cocoso_bonferroni_results.csv", mime="text/csv")
        except Exception as e:
            st.error(str(e))
