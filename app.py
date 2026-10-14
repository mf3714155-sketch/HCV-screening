import streamlit as st
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Hepatitis C Screening", page_icon="🩺")

THRESHOLD = 0.3

@st.cache_resource
def load_model():
    df = pd.read_csv("hcvdat0.csv")
    df = df.drop(columns=["Unnamed: 0"], errors="ignore")
    df["Sex"] = df["Sex"].map({"m": 0, "f": 1})
    df["target"] = df["Category"].astype(str).str[0].isin(["1", "2", "3"]).astype(int)
    X = df.drop(columns=["Category", "target"])
    y = df["target"]
    model = Pipeline([
        ("fill_missing", SimpleImputer(strategy="median")),
        ("rf", RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42)),
    ])
    model.fit(X, y)
    return model, list(X.columns)

model, columns = load_model()

defaults = {"Age": 45.0, "ALB": 41.0, "ALP": 68.0, "ALT": 28.0, "AST": 34.0,
            "BIL": 11.0, "CHE": 8.2, "CHOL": 5.4, "CREA": 81.0, "GGT": 39.0, "PROT": 72.0}

st.title("🩺 Hepatitis C Screening System")
st.write("Apni blood test report ki values neeche enter karein.")

values = {}
for col in columns:
    if col == "Sex":
        choice = st.selectbox("Sex", ["Male", "Female"])
        values[col] = 0 if choice == "Male" else 1
    else:
        values[col] = st.number_input(col, value=defaults.get(col, 0.0))

if st.button("Check karein"):
    row = pd.DataFrame([values], columns=columns)
    prob = model.predict_proba(row)[0, 1]
    st.write(f"Hepatitis C ka andaaza (probability): **{prob*100:.1f}%**")
    if prob >= THRESHOLD:
        st.error("⚠️ Hepatitis C ka khatra ho sakta hai. Baraye meherbani doctor se confirmatory test karwayein.")
    else:
        st.success("✅ Is waqt khatra kam nazar aata hai.")

st.caption("Ye sirf screening tool hai, diagnosis nahi. Doctor ka mashwara zaroori hai.")