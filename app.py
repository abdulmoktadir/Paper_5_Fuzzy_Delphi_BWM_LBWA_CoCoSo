import streamlit as st
import pandas as pd
import numpy as np
from modules import fuzzy_ops
from modules import fuzzy_delphi
from modules import fuzzy_bwm
from modules import fuzzy_lbwa
from modules import fuzzy_hybrid
from modules import fuzzy_cocoso

st.set_page_config(page_title="TS-to-Energy MCDM", layout="wide")
st.title("Integrated MCDM for Tannery Sludge-to-Energy Technology Selection")
st.markdown("Based on Moktadir et al. (2024), Chemical Engineering Journal")

# Sidebar navigation
module = st.sidebar.radio("Select Module", 
    ["Home", "1. Fuzzy Delphi", "2. Fuzzy BWM", "3. Fuzzy LBWA & Hybrid", "4. Fuzzy CoCoSo"])

# -------------------- Home --------------------
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

# -------------------- Module 1: Fuzzy Delphi --------------------
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
        # Sample from paper: Table S10 (simplified)
        sample_ratings = pd.read_csv("data/delphi_sample.csv")  # We'll provide a sample file
        st.dataframe(sample_ratings)
        ratings = sample_ratings.values.tolist()
    else:
        st.info("Enter expert ratings for each criterion.")
        # Create input table
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
        # ratings: list of experts, each expert is list of TFN per criterion
        # We need per criterion: list of TFNs from all experts
        crit_tfns = []
        for c in range(n_criteria):
            tfns = [ratings[e][c] for e in range(len(ratings))]
            crit_tfns.append(tfns)
        
        selected, gm_vals, gmi_vals = fuzzy_delphi.run_delphi(crit_tfns, threshold)
        
        st.subheader("Results")
        results_df = pd.DataFrame({
            "Criterion": [f"C{i+1}" for i in range(n_criteria)],
            "Aggregated TFN": [f"({gm[0]:.3f}, {gm[1]:.3f}, {gm[2]:.3f})" for gm in gm_vals],
            "GMI": [round(g, 4) for g in gmi_vals],
            "Selected": ["Yes" if s else "No" for s in selected]
        })
        st.dataframe(results_df)
        st.success(f"{sum(selected)} criteria selected (GMI ≥ {threshold})")

# -------------------- Module 2: Fuzzy BWM --------------------
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
    
    # Number of experts
    n_exp = st.number_input("Number of experts", min_value=1, value=3, step=1)
    
    # We'll assume all experts agree on best and worst for simplicity
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
        
        # Create data entry tables
        st.subheader("Best-to-Others Comparisons")
        best_to_others = []
        for e in range(n_exp):
            st.write(f"**Expert {e+1}**")
            row = []
            for j in range(len(factors)):
                if j == best_idx:
                    row.append((1,1,1))  # self comparison
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
            # Run BWM
            weights, xi, consistency = fuzzy_bwm.run_bwm(
                factors, best_idx, worst_idx, best_to_others, others_to_worst
            )
            st.subheader("Fuzzy Weights")
            for i, w in enumerate(weights):
                st.write(f"{factors[i]}: ({w[0]:.4f}, {w[1]:.4f}, {w[2]:.4f})  GMI = {fuzzy_ops.gmi(w):.4f}")
            st.info(f"Consistency index ξ = {xi:.4f}")
            st.session_state['bwm_weights'] = weights

# -------------------- Module 3: Fuzzy LBWA & Hybrid --------------------
elif module == "3. Fuzzy LBWA & Hybrid":
    st.header("Fuzzy LBWA and Hybrid Weighting")
    
    # Need factors and best factor from previous or input
    if st.checkbox("Use same factors as in BWM"):
        if 'bwm_factors' in st.session_state:
            factors = st.session_state['bwm_factors']
            best_idx = st.session_state['bwm_best']
        else:
            st.warning("Please run BWM first or input manually.")
            factors = ["T", "E", "En", "S", "G"]
            best_idx = 1  # Economic
    else:
        n_factors = st.number_input("Number of factors", min_value=2, value=5, step=1)
        factors = [st.text_input(f"Factor {i+1} name", value=f"F{i+1}") for i in range(n_factors)]
        best_idx = st.selectbox("Best factor", options=range(n_factors), format_func=lambda x: factors[x])
    
    n_exp = st.number_input("Number of experts", min_value=1, value=5, step=1)
    
    # Input level and ƛ for each factor (expert-wise)
    st.markdown("For each factor, assign a **level** (1,2,...) and a **comparison value ƛ** (0–τ).")
    st.markdown("τ = maximum number of factors in any level (auto‑computed).")
    
    # Create input tables
    level_data = []
    lambda_data = []  # per expert per factor
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
        # First compute LBWA weights
        lbwa_weights = fuzzy_lbwa.run_lbwa(factors, best_idx, level_data, lambda_data)
        st.subheader("Fuzzy LBWA Weights")
        for i, w in enumerate(lbwa_weights):
            st.write(f"{factors[i]}: ({w[0]:.4f}, {w[1]:.4f}, {w[2]:.4f})  GMI = {fuzzy_ops.gmi(w):.4f}")
        
        # Hybrid with BWM weights if available
        if 'bwm_weights' in st.session_state:
            bwm_weights = st.session_state['bwm_weights']
            hybrid_weights = fuzzy_hybrid.combine_weights(bwm_weights, lbwa_weights, alpha_tfn, beta_tfn)
            st.subheader("Hybrid (FBWM + FLBWA) Weights")
            for i, w in enumerate(hybrid_weights):
                st.write(f"{factors[i]}: ({w[0]:.4f}, {w[1]:.4f}, {w[2]:.4f})  GMI = {fuzzy_ops.gmi(w):.4f}")
            st.session_state['hybrid_weights'] = hybrid_weights
        else:
            st.warning("BWM weights not found. Using LBWA only.")
            st.session_state['hybrid_weights'] = lbwa_weights

# -------------------- Module 4: Fuzzy CoCoSo with Bonferroni --------------------
elif module == "4. Fuzzy CoCoSo":
    st.header("Fuzzy CoCoSo with Bonferroni – Technology Ranking")
    
    # Input alternatives
    alt_names = ["TS-SS", "TS-HP", "TS-PCC", "TS-MPP"] if st.checkbox("Use sample alternatives") else \
                st.text_area("Alternative names (one per line)").splitlines()
    
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
            l = st.number_input(f"Weight {criteria_names[i]} (l)", value=0.1, key=f"w{i}_l")
            m = st.number_input(f"Weight {criteria_names[i]} (m)", value=0.2, key=f"w{i}_m")
            u = st.number_input(f"Weight {criteria_names[i]} (u)", value=0.3, key=f"w{i}_u")
            weights.append((l, m, u))
    
    # Criteria types
    types = []
    for i, c in enumerate(criteria_names):
        typ = st.selectbox(f"Type of {c}", ["benefit", "cost"], key=f"type_{i}")
        types.append(typ)
    
    # Decision matrix: for simplicity, we provide sample data or allow upload
    if st.checkbox("Use sample decision matrix (from paper)"):
        # Sample aggregated matrix from Table S45
        sample_matrix = pd.read_csv("data/cocoso_sample.csv")
        st.dataframe(sample_matrix)
        # Convert to list of lists of TFNs
        decision = []
        for idx, row in sample_matrix.iterrows():
            alt = []
            for j, c in enumerate(criteria_names):
                l = row[f"{c}_l"]
                m = row[f"{c}_m"]
                u = row[f"{c}_u"]
                alt.append((l, m, u))
            decision.append(alt)
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
    
    if st.button("Run Fuzzy CoCoSo"):
        # Normalize
        norm_matrix = fuzzy_cocoso.normalize(decision, types)
        # Compute weighted sequences
        scob, pcob = fuzzy_cocoso.compute_bonferroni(norm_matrix, weights, phi1, phi2)
        # Compute relative significance
        psi_a, psi_b, psi_c = fuzzy_cocoso.relative_significance(scob, pcob, pi)
        # Final scores
        final_scores = fuzzy_cocoso.final_scores(psi_a, psi_b, psi_c)
        # Rank
        rankings = sorted([(i, fuzzy_ops.gmi(final_scores[i])) for i in range(len(alt_names))], 
                          key=lambda x: x[1], reverse=True)
        
        st.subheader("Ranking Results")
        for rank, (idx, score) in enumerate(rankings, start=1):
            st.write(f"{rank}. {alt_names[idx]} – GMI = {score:.4f}")
