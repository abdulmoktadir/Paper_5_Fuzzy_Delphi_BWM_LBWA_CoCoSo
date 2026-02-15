# Integrated Fuzzy MCDM Streamlit App

This app implements the 4-module workflow described in the attached paper:

1) Fuzzy Delphi  
2) Fuzzy BWM (3-LP practical fuzzy extension)  
3) Fuzzy LBWA + Hybrid weight aggregation (non-linear operator)  
4) Fuzzy Bonferroni CoCoSo (CoCoSo-B) with GMI defuzzification

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Notes

- You can input TFNs directly as strings like `(0.1,0.3,0.5)` in Module 4.
- Module 4 converts numeric values to TFNs using uncertainty `σ`: `(x(1-σ), x, x(1+σ))`.
