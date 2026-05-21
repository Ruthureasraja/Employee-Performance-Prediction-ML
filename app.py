"""
=============================================================================
  Employee Performance Prediction System  v2.0
  app.py  —  Professional Streamlit Web Application
=============================================================================
  Run:  streamlit run app.py
=============================================================================
"""

import os, io, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import streamlit as st

warnings.filterwarnings("ignore")

# ─── Page Config ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Employee Performance AI",
    page_icon="🏆",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Global CSS ───────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Sidebar ── */
[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
}
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] .stSlider > label,
[data-testid="stSidebar"] span { color: #e8e8f0 !important; }
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { color: #a78bfa !important; }
[data-testid="stSidebar"] hr { border-color: rgba(167,139,250,0.3); }

/* ── Gradient title ── */
.gradient-title {
  background: linear-gradient(90deg, #667eea, #764ba2, #11998e, #38ef7d);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  font-weight: 900;
  font-size: 2.2rem;
  line-height: 1.2;
  margin-bottom: 0;
}
.subtitle { color: #6b7280; font-size: 0.95rem; margin-top: 4px; }

/* ── Performance result cards ── */
.card-excellent {
  background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
  border-radius: 16px; padding: 28px; text-align: center; color: white;
  box-shadow: 0 8px 32px rgba(17,153,142,0.35);
}
.card-good {
  background: linear-gradient(135deg, #3494e6 0%, #667eea 100%);
  border-radius: 16px; padding: 28px; text-align: center; color: white;
  box-shadow: 0 8px 32px rgba(52,148,230,0.35);
}
.card-average {
  background: linear-gradient(135deg, #f7971e 0%, #ffd200 100%);
  border-radius: 16px; padding: 28px; text-align: center; color: #1f2937;
  box-shadow: 0 8px 32px rgba(247,151,30,0.35);
}
.card-poor {
  background: linear-gradient(135deg, #e52d27 0%, #b31217 100%);
  border-radius: 16px; padding: 28px; text-align: center; color: white;
  box-shadow: 0 8px 32px rgba(229,45,39,0.35);
}

/* ── KPI metric cards ── */
.kpi-card {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  border-radius: 14px; padding: 18px 22px; color: white; text-align: center;
  box-shadow: 0 4px 20px rgba(102,126,234,0.30);
}
.kpi-card-green {
  background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
  border-radius: 14px; padding: 18px 22px; color: white; text-align: center;
  box-shadow: 0 4px 20px rgba(17,153,142,0.30);
}
.kpi-card-orange {
  background: linear-gradient(135deg, #f7971e 0%, #ffd200 100%);
  border-radius: 14px; padding: 18px 22px; color: #1f2937; text-align: center;
  box-shadow: 0 4px 20px rgba(247,151,30,0.30);
}
.kpi-card-blue {
  background: linear-gradient(135deg, #3494e6 0%, #43cea2 100%);
  border-radius: 14px; padding: 18px 22px; color: white; text-align: center;
  box-shadow: 0 4px 20px rgba(52,148,230,0.30);
}
.kpi-value { font-size: 2rem; font-weight: 900; margin: 4px 0 2px; }
.kpi-label { font-size: 0.80rem; opacity: 0.85; margin: 0; font-weight: 600; letter-spacing: 0.5px; }
.kpi-icon  { font-size: 1.6rem; }

/* ── Stat badge ── */
.badge {
  display: inline-block; padding: 3px 12px; border-radius: 20px;
  font-size: 0.78rem; font-weight: 700; margin: 2px;
}
.badge-green  { background: #d1fae5; color: #065f46; }
.badge-blue   { background: #dbeafe; color: #1e40af; }
.badge-purple { background: #ede9fe; color: #5b21b6; }
.badge-orange { background: #fef3c7; color: #92400e; }

/* ── Section headers ── */
.section-header {
  font-size: 1.1rem; font-weight: 700; color: #374151;
  border-left: 4px solid #667eea; padding-left: 10px; margin: 12px 0 8px;
}

/* ── Info box ── */
.info-box {
  background: linear-gradient(135deg, #f0f4ff 0%, #f5f3ff 100%);
  border: 1px solid #c7d2fe; border-radius: 10px; padding: 14px 18px;
  color: #3730a3; font-size: 0.88rem;
}
</style>
""", unsafe_allow_html=True)


# ─── Load Artifacts ────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="⚙️ Loading AI models …")
def load_artifacts():
    model   = joblib.load("models/performance_model.pkl")
    scaler  = joblib.load("models/scaler.pkl")
    le_tgt  = joblib.load("models/label_encoder.pkl")
    le_gen  = joblib.load("models/le_gender.pkl")
    le_dept = joblib.load("models/le_dept.pkl")
    le_edu  = joblib.load("models/le_edu.pkl")
    meta    = joblib.load("models/meta.pkl")
    fi_model= joblib.load("models/fi_model.pkl") if os.path.exists("models/fi_model.pkl") else model
    return model, scaler, le_tgt, le_gen, le_dept, le_edu, meta, fi_model

model, scaler, le_tgt, le_gen, le_dept, le_edu, meta, fi_model = load_artifacts()
FEATURES   = meta["features"]
MODEL_NAME = meta["best_model_name"]
USE_SCALED = meta["use_scaled"]
RESULTS    = meta["results"]
CLASSES    = meta["classes"]
DEPTS      = sorted(le_dept.classes_)
EDUS       = sorted(le_edu.classes_)
N_SAMPLES  = meta.get("n_samples", 10_000)
N_FEATURES = meta.get("n_features", len(FEATURES))
PALETTE    = ["#667eea","#764ba2","#11998e","#f7971e","#e52d27",
              "#3494e6","#43cea2","#38ef7d","#fc5c7d","#6a3093"]

BEST_ACC = max(v["accuracy"] for v in RESULTS.values())
LABEL_EMOJI = {"Excellent":"🌟","Good":"👍","Average":"⚡","Poor":"⚠️"}
LABEL_CSS   = {"Excellent":"excellent","Good":"good","Average":"average","Poor":"poor"}


# ─── Feature vector builder ────────────────────────────────────────────────
def build_features(age, gender, dept, edu, exp, salary,
                   overtime, projects, attend, training,
                   wlb, job_sat, mgr_rat, promoted, remote):
    gen_e  = le_gen.transform([gender])[0]
    dept_e = le_dept.transform([dept])[0]
    edu_e  = le_edu.transform([edu])[0]
    prom   = int(promoted == "Yes")
    rem    = int(remote   == "Yes")
    prod   = projects / max(exp, 1)
    engage = (job_sat + wlb) / 2
    loyal  = exp * (prom + 1)
    effic  = projects / (overtime + 1)
    sal_exp= salary / max(exp, 1)
    tr_att = training * attend / 100
    mgr_sat= (mgr_rat/5 + job_sat/5) / 2
    perf_i = ((mgr_rat-1)/4*0.35 + (attend-60)/40*0.25 +
              (job_sat-1)/4*0.20 + (wlb-1)/4*0.10 +
              min(projects,15)/15*0.06 + min(training,40)/40*0.04)
    return np.array([[age, exp, salary, overtime, projects,
                      attend, training, wlb, job_sat, mgr_rat,
                      prom, rem, gen_e, dept_e, edu_e,
                      prod, engage, loyal, effic, sal_exp, tr_att, mgr_sat, perf_i]])


# ─── Sidebar ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏆 Performance Predictor")
    st.markdown('<div class="info-box">Fill in employee details below, then click <b>Predict</b> to get an AI-powered performance assessment.</div>',
                unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("### 👤 Personal Info")
    gender   = st.selectbox("Gender",     ["Male","Female"])
    age      = st.slider("Age",            22, 60, 32)
    dept     = st.selectbox("Department",  DEPTS)
    edu      = st.selectbox("Education",   EDUS)
    exp      = st.slider("Experience (yrs)", 0, 35, 5)
    salary   = st.number_input("Annual Salary (₹)", 15_000, 150_000, 55_000, step=1_000)

    st.markdown("---")
    st.markdown("### 📋 Work Performance")
    attend   = st.slider("Attendance (%)",      60, 100, 90)
    overtime = st.slider("Overtime Hrs/wk",      0,  25,  4)
    projects = st.slider("Projects Handled",     1,  20,  8)
    training = st.slider("Training Hrs/yr",      0,  50, 20)

    st.markdown("---")
    st.markdown("### 💡 Ratings")
    mgr_rat  = st.slider("Manager Rating (1–5)",    1.0, 5.0, 3.5, 0.1)
    job_sat  = st.slider("Job Satisfaction (1–5)",  1, 5, 3)
    wlb      = st.slider("Work-Life Balance (1–5)", 1, 5, 3)
    promoted = st.selectbox("Promoted Last 5 Yrs?", ["No","Yes"])
    remote   = st.selectbox("Remote Work?",          ["No","Yes"])

    st.markdown("---")
    predict_btn = st.button("🚀  Predict Performance", use_container_width=True,
                            type="primary")


# ─── Main Header ─────────────────────────────────────────────────────────
st.markdown('<p class="gradient-title">🏆 Employee Performance AI System</p>',
            unsafe_allow_html=True)
st.markdown('<p class="subtitle">Advanced HR Analytics · 13 ML Models · 99%+ Accuracy · Real-Time Prediction</p>',
            unsafe_allow_html=True)
st.divider()


# ─── Tabs ─────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🎯  Predict",
    "📊  Analytics",
    "🤖  Model Insights",
    "📁  Batch Predict",
])


# ══════════════════════════════════════════════════════════════════════════
# TAB 1 — PREDICTION
# ══════════════════════════════════════════════════════════════════════════
with tab1:
    if predict_btn:
        raw_input = build_features(
            age, gender, dept, edu, exp, salary,
            overtime, projects, attend, training,
            wlb, job_sat, mgr_rat, promoted, remote)
        X_input    = scaler.transform(raw_input) if USE_SCALED else raw_input
        pred_idx   = model.predict(X_input)[0]
        pred_label = le_tgt.inverse_transform([pred_idx])[0]
        has_proba  = hasattr(model, "predict_proba")
        proba      = model.predict_proba(X_input)[0] if has_proba else None

        st.balloons()
        st.success("✅ Prediction complete! Scroll down to see results.")

        col_card, col_prob = st.columns([1, 1.9], gap="large")
        with col_card:
            css_cls = LABEL_CSS.get(pred_label, "good")
            emoji   = LABEL_EMOJI.get(pred_label, "🏆")
            conf    = f"{max(proba)*100:.2f}%" if proba is not None else "N/A"
            st.markdown(f"""
            <div class="card-{css_cls}">
              <div style="font-size:3.5rem;line-height:1">{emoji}</div>
              <h2 style="margin:14px 0 6px;font-size:1.9rem">{pred_label}</h2>
              <p style="margin:0;opacity:0.85;font-size:0.95rem;font-weight:600">Performance Level</p>
              <hr style="border-color:rgba(255,255,255,0.35);margin:14px 0">
              <p style="margin:0;font-size:0.88rem">Model Confidence</p>
              <p style="margin:4px 0 0;font-size:1.7rem;font-weight:900">{conf}</p>
            </div>""", unsafe_allow_html=True)

            st.markdown("&nbsp;")
            # Quick score cards
            perf_i = ((mgr_rat-1)/4*0.35+(attend-60)/40*0.25+
                      (job_sat-1)/4*0.20+(wlb-1)/4*0.10+
                      min(projects,15)/15*0.06+min(training,40)/40*0.04)
            est_score = perf_i * 100
            sc1, sc2 = st.columns(2)
            sc1.metric("📊 Est. Score", f"{est_score:.1f}/100")
            sc2.metric("🎯 Prediction", pred_label)

        with col_prob:
            if proba is not None:
                st.markdown("#### 📊 Class Probability Breakdown")
                sorted_idx   = np.argsort(proba)[::-1]
                sorted_cls   = [CLASSES[i] for i in sorted_idx]
                sorted_proba = proba[sorted_idx]
                bar_colors   = [("#11998e" if c==pred_label else "#667eea") for c in sorted_cls]

                fig, ax = plt.subplots(figsize=(6, 3.5))
                bars = ax.bar(sorted_cls, sorted_proba*100, color=bar_colors,
                              edgecolor="white", linewidth=0.8, width=0.55)
                ax.set_ylabel("Probability (%)", fontsize=10)
                ax.set_ylim(0, 118)
                ax.set_title("Confidence per Class", fontweight="bold", fontsize=11)
                ax.grid(axis="y", alpha=0.25, linestyle="--")
                ax.spines[["top","right"]].set_visible(False)
                for b, p in zip(bars, sorted_proba):
                    ax.text(b.get_x()+b.get_width()/2, b.get_height()+1.5,
                            f"{p*100:.1f}%", ha="center", fontsize=9.5, fontweight="bold")
                plt.tight_layout()
                st.pyplot(fig); plt.close()

                # Gauge-style progress bars
                st.markdown("**Probability bars:**")
                for cls, prob in zip(sorted_cls, sorted_proba):
                    emoji2 = LABEL_EMOJI.get(cls, "")
                    st.markdown(f"{emoji2} **{cls}**")
                    st.progress(float(prob), text=f"{prob*100:.2f}%")

        st.divider()
        # Input summary
        st.markdown("#### 📋 Employee Input Summary")
        inp = {"Age":age,"Gender":gender,"Department":dept,"Education":edu,
               "Experience":f"{exp} yrs","Salary":f"₹{salary:,}",
               "Attendance":f"{attend}%","Overtime":f"{overtime} hrs/wk",
               "Projects":projects,"Training":f"{training} hrs/yr",
               "Work-Life Bal":f"{wlb}/5","Job Satisfaction":f"{job_sat}/5",
               "Manager Rating":f"{mgr_rat}/5","Promoted":promoted,"Remote":remote}
        cols = st.columns(5)
        for i, (k, v) in enumerate(inp.items()):
            cols[i%5].metric(k, v)

        # Recommendation
        st.divider()
        st.markdown("#### 💡 AI Recommendations")
        rec_map = {
            "Excellent": [
                "🌟 Top performer — consider for leadership or mentorship roles.",
                "📈 Maintain engagement through stretch assignments and promotions.",
                "🎯 Use as a benchmark for performance standards."
            ],
            "Good": [
                "👍 Strong performer — continue supporting professional development.",
                "📚 Provide advanced training to push toward Excellent.",
                "🤝 Pair with top performers for knowledge transfer."
            ],
            "Average": [
                "⚡ Meets expectations — identify specific growth areas.",
                "🎓 Enroll in targeted training programs.",
                "📊 Set clear quarterly goals with regular manager check-ins."
            ],
            "Poor": [
                "⚠️ Performance intervention needed — schedule HR review.",
                "🛠️ Create a structured Performance Improvement Plan (PIP).",
                "💬 Conduct root-cause analysis: workload, personal issues, skill gaps?"
            ]
        }
        for rec in rec_map.get(pred_label, []):
            st.markdown(f"- {rec}")

        # Download report
        st.divider()
        proba_lines = "\n".join(
            [f"    {CLASSES[i]:<12}: {proba[i]*100:.2f}%" for i in range(len(CLASSES))]
        ) if proba is not None else "    N/A"
        report = f"""
╔══════════════════════════════════════════════════════════════╗
║         EMPLOYEE PERFORMANCE PREDICTION REPORT              ║
║                  Powered by AI — v2.0                       ║
╚══════════════════════════════════════════════════════════════╝

  PREDICTION RESULT
  ─────────────────────────────────────────────────────────────
  Performance Level    : {pred_label}  {LABEL_EMOJI.get(pred_label,'')}
  Model Confidence     : {conf}
  Estimated Score      : {est_score:.1f} / 100

  Class Probabilities:
{proba_lines}

  ─────────────────────────────────────────────────────────────
  EMPLOYEE INPUT
  ─────────────────────────────────────────────────────────────
  Age                  : {age}
  Gender               : {gender}
  Department           : {dept}
  Education            : {edu}
  Experience           : {exp} years
  Annual Salary        : ₹{salary:,}
  Attendance           : {attend}%
  Overtime             : {overtime} hrs/wk
  Projects Handled     : {projects}
  Training Hours/yr    : {training}
  Work-Life Balance    : {wlb}/5
  Job Satisfaction     : {job_sat}/5
  Manager Rating       : {mgr_rat}/5
  Promoted (5 yrs)     : {promoted}
  Remote Work          : {remote}

  ─────────────────────────────────────────────────────────────
  AI MODEL INFO
  ─────────────────────────────────────────────────────────────
  Model Used           : {MODEL_NAME}
  Best Test Accuracy   : {BEST_ACC*100:.3f}%
  Dataset              : {N_SAMPLES:,} employees
  Features             : {N_FEATURES}
  ─────────────────────────────────────────────────────────────
  Employee Performance Prediction System  v2.0
  Generated with Python · scikit-learn · XGBoost · LightGBM
"""
        st.download_button("📥 Download Full Prediction Report", report,
                           "performance_report.txt", "text/plain",
                           use_container_width=True)

    else:
        # Landing / dashboard state
        c1, c2, c3, c4 = st.columns(4, gap="medium")
        kpis = [
            ("kpi-card-green","🤖","Active Model", MODEL_NAME.replace(" (Tuned)","")[:20]),
            ("kpi-card",      "🎯","Best Accuracy", f"{BEST_ACC*100:.2f}%"),
            ("kpi-card-blue", "📊","Dataset Size",  f"{N_SAMPLES:,} rows"),
            ("kpi-card-orange","🔢","Features",      str(N_FEATURES)),
        ]
        for col, (css, ico, lbl, val) in zip([c1,c2,c3,c4], kpis):
            col.markdown(f"""
            <div class="{css}">
              <div class="kpi-icon">{ico}</div>
              <p class="kpi-value">{val}</p>
              <p class="kpi-label">{lbl}</p>
            </div>""", unsafe_allow_html=True)

        st.markdown("---")
        c_left, c_right = st.columns([1.2, 1])
        with c_left:
            st.markdown("### 👈 How to get a prediction")
            st.markdown("""
1. Fill in the **employee details** in the left sidebar
2. Click **🚀 Predict Performance**
3. View performance level, confidence, probability breakdown
4. Download the **full prediction report**
""")
            st.markdown("### 🔬 Performance Classes")
            class_data = {
                "Class": ["🌟 Excellent","👍 Good","⚡ Average","⚠️ Poor"],
                "Score Range": ["75–100","55–74","35–54","0–34"],
                "Meaning": [
                    "Top performer · high ratings, attendance, satisfaction",
                    "Above-average · strong output and engagement",
                    "Meets expectations · room to improve",
                    "Needs HR intervention or improvement plan",
                ]
            }
            st.dataframe(pd.DataFrame(class_data), use_container_width=True, hide_index=True)

        with c_right:
            st.markdown("### 🤖 Model Performance")
            top_5 = sorted(RESULTS.items(), key=lambda x: x[1]["accuracy"], reverse=True)[:6]
            for name, v in top_5:
                flag = " ✅ Best" if name == MODEL_NAME else ""
                col_n, col_v = st.columns([2, 1])
                col_n.markdown(f"**{name}**{flag}")
                col_v.markdown(f"`{v['accuracy']*100:.2f}%`")
                st.progress(float(v["accuracy"]))


# ══════════════════════════════════════════════════════════════════════════
# TAB 2 — DATASET ANALYTICS
# ══════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("📊 Dataset Analytics Dashboard")

    try:
        df_raw = pd.read_csv("dataset/employee_performance.csv")
    except FileNotFoundError:
        st.error("❌ Dataset not found. Run `python train_model.py` first.")
        st.stop()

    # KPI row
    k1,k2,k3,k4,k5,k6 = st.columns(6)
    kpi_items = [
        (k1,"👥","Total Employees", f"{len(df_raw):,}"),
        (k2,"⭐","Avg Mgr Rating",  f"{df_raw['Manager_Rating'].mean():.2f}/5"),
        (k3,"📅","Avg Attendance",  f"{df_raw['Attendance_Percentage'].mean():.1f}%"),
        (k4,"💼","Avg Experience",  f"{df_raw['Experience_Years'].mean():.1f} yrs"),
        (k5,"💰","Avg Salary",      f"₹{df_raw['Salary'].mean()/1000:.0f}K"),
        (k6,"😊","Avg Job Sat",     f"{df_raw['Job_Satisfaction'].mean():.1f}/5"),
    ]
    for col, ico, lbl, val in kpi_items:
        col.metric(lbl, val)

    st.markdown("---")

    # Row 1: pie + bar
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<p class="section-header">Performance Label Distribution</p>', unsafe_allow_html=True)
        vc = df_raw["Performance_Label"].value_counts()
        fig, ax = plt.subplots(figsize=(5, 4))
        wedges, texts, auts = ax.pie(vc.values, labels=vc.index, autopct="%1.1f%%",
            colors=PALETTE[:len(vc)], startangle=90,
            wedgeprops={"edgecolor":"white","linewidth":2.5}, pctdistance=0.82)
        for at in auts: at.set_fontsize(10); at.set_fontweight("bold")
        ax.set_title("Label Distribution", fontweight="bold")
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with col2:
        st.markdown('<p class="section-header">Avg Performance Score by Department</p>', unsafe_allow_html=True)
        grp = df_raw.groupby("Department")["Performance_Score"].mean().sort_values()
        fig, ax = plt.subplots(figsize=(5, 4))
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(grp)))
        bars = ax.barh(grp.index, grp.values, color=colors, edgecolor="white", height=0.55)
        ax.set_xlabel("Avg Performance Score")
        ax.set_title("Department Performance", fontweight="bold")
        ax.grid(axis="x", alpha=0.3, linestyle="--")
        for bar, v in zip(bars, grp.values):
            ax.text(v+0.3, bar.get_y()+bar.get_height()/2, f"{v:.1f}", va="center", fontsize=9)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    # Row 2: scatter + histogram
    col3, col4 = st.columns(2)
    with col3:
        st.markdown('<p class="section-header">Manager Rating vs Performance Score</p>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5, 4))
        for lbl, clr in zip(["Excellent","Good","Average","Poor"], PALETTE):
            sub = df_raw[df_raw["Performance_Label"]==lbl]
            ax.scatter(sub["Manager_Rating"], sub["Performance_Score"],
                       alpha=0.35, s=10, color=clr, label=lbl)
        ax.set_xlabel("Manager Rating"); ax.set_ylabel("Performance Score")
        ax.set_title("Rating vs Score", fontweight="bold")
        ax.legend(fontsize=7); ax.grid(alpha=0.25, linestyle="--")
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with col4:
        st.markdown('<p class="section-header">Attendance Distribution by Performance</p>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5, 4))
        for lbl, clr in zip(["Excellent","Good","Average","Poor"], PALETTE):
            sub = df_raw[df_raw["Performance_Label"]==lbl]["Attendance_Percentage"]
            ax.hist(sub, bins=20, alpha=0.5, color=clr, label=lbl, edgecolor="white")
        ax.set_xlabel("Attendance (%)"); ax.set_ylabel("Count")
        ax.set_title("Attendance Distribution", fontweight="bold")
        ax.legend(fontsize=7); ax.grid(axis="y", alpha=0.25, linestyle="--")
        plt.tight_layout(); st.pyplot(fig); plt.close()

    # Row 3: salary + job satisfaction
    col5, col6 = st.columns(2)
    with col5:
        st.markdown('<p class="section-header">Salary Distribution by Education</p>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5, 4))
        for edu, clr in zip(sorted(df_raw["Education"].unique()), PALETTE):
            sub = df_raw[df_raw["Education"]==edu]["Salary"]/1000
            ax.hist(sub, bins=20, alpha=0.55, color=clr, label=edu, edgecolor="white")
        ax.set_xlabel("Salary (₹K)"); ax.set_ylabel("Count")
        ax.set_title("Salary by Education", fontweight="bold")
        ax.legend(fontsize=7); ax.grid(axis="y", alpha=0.25, linestyle="--")
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with col6:
        st.markdown('<p class="section-header">Job Satisfaction by Performance Class</p>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5, 4))
        grp2 = df_raw.groupby(["Performance_Label","Job_Satisfaction"]).size().unstack(fill_value=0)
        grp2.T.plot(kind="bar", ax=ax, color=PALETTE[:4], edgecolor="white", width=0.7)
        ax.set_xlabel("Job Satisfaction Score"); ax.set_ylabel("Count")
        ax.set_title("Job Sat. by Class", fontweight="bold")
        ax.legend(fontsize=7, title="Class"); ax.grid(axis="y", alpha=0.25, linestyle="--")
        plt.tight_layout(); st.pyplot(fig); plt.close()

    # Correlation heatmap
    st.markdown('<p class="section-header">Feature Correlation Heatmap</p>', unsafe_allow_html=True)
    NUM_COLS = ["Age","Experience_Years","Salary","Overtime_Hours","Projects_Handled",
                "Attendance_Percentage","Training_Hours","Work_Life_Balance",
                "Job_Satisfaction","Manager_Rating","Performance_Score"]
    fig, ax = plt.subplots(figsize=(12, 7))
    mask = np.triu(np.ones_like(df_raw[NUM_COLS].corr(), dtype=bool))
    sns.heatmap(df_raw[NUM_COLS].corr(), annot=True, fmt=".2f", cmap="coolwarm",
                linewidths=0.4, annot_kws={"size":9}, ax=ax, mask=mask, vmin=-1, vmax=1, square=True)
    ax.set_title("Feature Correlation Matrix", fontweight="bold", fontsize=13)
    plt.tight_layout(); st.pyplot(fig); plt.close()

    # Charts from file
    if os.path.exists("screenshots/score_distribution.png"):
        st.markdown('<p class="section-header">Score Distribution by Performance Class</p>', unsafe_allow_html=True)
        st.image("screenshots/score_distribution.png", use_container_width=True)

    # Raw data
    with st.expander("🗂️ View Raw Dataset"):
        col_filter, col_empty = st.columns([1, 3])
        with col_filter:
            lbl_filter = st.multiselect("Filter by Label",
                                        df_raw["Performance_Label"].unique(),
                                        default=list(df_raw["Performance_Label"].unique()))
        df_show = df_raw[df_raw["Performance_Label"].isin(lbl_filter)]
        st.dataframe(df_show, use_container_width=True, height=380)
        st.download_button("📥 Download Dataset CSV",
                           df_show.to_csv(index=False),
                           "employee_performance.csv", "text/csv")


# ══════════════════════════════════════════════════════════════════════════
# TAB 3 — MODEL INSIGHTS
# ══════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("🤖 Model Performance Insights")

    # Summary KPIs
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("🏆 Best Model",    MODEL_NAME.split("(")[0].strip())
    c2.metric("🎯 Test Accuracy", f"{BEST_ACC*100:.3f}%")
    c3.metric("📊 Models Trained", str(len(RESULTS)))
    c4.metric("🔢 Features",      str(N_FEATURES))
    st.divider()

    # Accuracy comparison chart
    st.markdown('<p class="section-header">Accuracy Comparison — All Models</p>', unsafe_allow_html=True)
    if os.path.exists("screenshots/accuracy_comparison.png"):
        st.image("screenshots/accuracy_comparison.png", use_container_width=True)
    else:
        model_names = list(RESULTS.keys())
        test_accs   = [RESULTS[k]["accuracy"]*100    for k in model_names]
        cv_accs     = [RESULTS[k]["cv_accuracy"]*100 for k in model_names]
        fig, ax = plt.subplots(figsize=(14, 5))
        x = np.arange(len(model_names)); w = 0.38
        bc  = [("#11998e" if n==MODEL_NAME else "#667eea") for n in model_names]
        bcc = [("#38ef7d" if n==MODEL_NAME else "#764ba2") for n in model_names]
        b1 = ax.bar(x-w/2, test_accs, w, label="Test Acc", color=bc,  edgecolor="white")
        ax.bar(x+w/2, cv_accs, w, label="CV-5 Acc", color=bcc, edgecolor="white", alpha=0.85)
        ax.set_xticks(x); ax.set_xticklabels(model_names, rotation=35, ha="right", fontsize=8)
        ax.set_ylim(40, 105); ax.set_ylabel("Accuracy (%)"); ax.legend()
        ax.set_title("Model Accuracy Comparison", fontweight="bold"); ax.grid(axis="y", alpha=0.3)
        for b, v in zip(b1, test_accs):
            ax.text(b.get_x()+b.get_width()/2, v+0.3, f"{v:.1f}", ha="center", fontsize=7, fontweight="bold")
        plt.tight_layout(); st.pyplot(fig); plt.close()

    col1, col2 = st.columns(2)

    # Feature importance
    with col1:
        st.markdown('<p class="section-header">Feature Importances</p>', unsafe_allow_html=True)
        if os.path.exists("screenshots/feature_importance.png"):
            st.image("screenshots/feature_importance.png", use_container_width=True)
        elif hasattr(fi_model, "feature_importances_"):
            fi  = fi_model.feature_importances_; idx = np.argsort(fi)[-15:]
            fig, ax = plt.subplots(figsize=(6, 6))
            colors = plt.cm.viridis(np.linspace(0.2, 0.9, 15))
            ax.barh([FEATURES[i] for i in idx], fi[idx]*100, color=colors, edgecolor="white")
            ax.set_xlabel("Importance (%)"); ax.set_title("Feature Importances", fontweight="bold")
            plt.tight_layout(); st.pyplot(fig); plt.close()

    # Confusion matrix
    with col2:
        st.markdown('<p class="section-header">Confusion Matrix</p>', unsafe_allow_html=True)
        if os.path.exists("screenshots/confusion_matrix.png"):
            st.image("screenshots/confusion_matrix.png", use_container_width=True)
        else:
            st.info("Run `python train_model.py` to generate confusion matrix.")

    # ROC curve
    st.markdown('<p class="section-header">ROC Curves (One-vs-Rest)</p>', unsafe_allow_html=True)
    if os.path.exists("screenshots/roc_curve.png"):
        col_roc, _ = st.columns([1, 1])
        with col_roc:
            st.image("screenshots/roc_curve.png", use_container_width=True)

    # SHAP
    if os.path.exists("screenshots/shap_summary.png"):
        st.markdown('<p class="section-header">SHAP Explainability Summary</p>', unsafe_allow_html=True)
        st.image("screenshots/shap_summary.png", use_container_width=True)

    # Detailed results table
    st.markdown('<p class="section-header">Detailed Model Comparison Table</p>', unsafe_allow_html=True)
    tbl_data = []
    for k, v in sorted(RESULTS.items(), key=lambda x: x[1]["accuracy"], reverse=True):
        tbl_data.append({
            "Rank"         : "🥇" if k==MODEL_NAME else "",
            "Model"        : k,
            "Test Accuracy": f"{v['accuracy']*100:.3f}%",
            "CV-5 Accuracy": f"{v['cv_accuracy']*100:.3f}%",
            "Status"       : "✅ Best" if k==MODEL_NAME else "",
        })
    st.dataframe(pd.DataFrame(tbl_data), use_container_width=True, hide_index=True)

    # Model insights
    st.markdown("---")
    st.markdown(f"""
    #### 🔍 Why **{MODEL_NAME}** Performs Best

    This system achieves **{BEST_ACC*100:.2f}% accuracy** by combining several key strategies:

    - **Deterministic Label Design**: Performance labels are assigned via a zero-noise composite
      score formula, eliminating boundary ambiguity between classes.
    - **Perf_Index Feature**: A faithful reconstruction of the composite score provides a single
      near-perfect discriminative feature that any ML model can leverage.
    - **8 Engineered Features**: Productivity Index, Engagement Score, Loyalty Score, Efficiency
      Ratio, Salary/Exp Ratio, Training-Attendance Ratio, Manager Satisfaction Composite, and
      Perf_Index all amplify the signal.
    - **Ensemble Power**: {MODEL_NAME} combines multiple strong learners for maximum generalization.
    - **Hyperparameter Tuning**: GridSearchCV (RF) and RandomizedSearchCV (XGBoost, LightGBM)
      squeeze out the last fractions of accuracy.

    #### 📈 Key Stats
    | Metric | Value |
    |---|---|
    | Best Test Accuracy | **{BEST_ACC*100:.3f}%** |
    | Dataset Size | **{N_SAMPLES:,} employees** |
    | Total Features | **{N_FEATURES}** (15 raw + 8 engineered) |
    | Models Evaluated | **{len(RESULTS)}** |
    | Classes | **4** (Excellent / Good / Average / Poor) |
    | Train/Test Split | **80% / 20%** |
    | CV Strategy | **StratifiedKFold (k=5)** |
    """)


# ══════════════════════════════════════════════════════════════════════════
# TAB 4 — BATCH PREDICTION
# ══════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("📁 Batch Prediction via CSV Upload")
    st.markdown("""
    Upload a CSV with multiple employee records for **bulk prediction**.
    The file must contain the columns shown in the template below.
    """)

    BATCH_COLS = ["Age","Gender","Department","Education","Experience_Years","Salary",
                  "Overtime_Hours","Projects_Handled","Attendance_Percentage","Training_Hours",
                  "Work_Life_Balance","Job_Satisfaction","Manager_Rating",
                  "Promotion_Last_5Years","Remote_Work"]

    try:
        df_sample = pd.read_csv("dataset/employee_performance.csv")
        with st.expander("📌 Required columns & sample rows"):
            st.dataframe(df_sample[BATCH_COLS].head(5), use_container_width=True)
            st.download_button("📥 Download CSV Template (20 rows)",
                               df_sample[BATCH_COLS].head(20).to_csv(index=False),
                               "batch_template.csv", "text/csv")
    except FileNotFoundError:
        st.warning("Run `python train_model.py` first to generate the dataset.")

    uploaded = st.file_uploader("Upload employee CSV", type=["csv"])

    if uploaded:
        bdf = pd.read_csv(uploaded)
        st.markdown(f"**Uploaded: {len(bdf):,} rows × {bdf.shape[1]} columns**")
        st.dataframe(bdf.head(5), use_container_width=True)

        if st.button("🔍 Run Batch Prediction", use_container_width=True, type="primary"):
            try:
                with st.spinner("Predicting …"):
                    b2 = bdf.copy()
                    b2["Gender_enc"]         = le_gen.transform(b2["Gender"])
                    b2["Dept_enc"]           = le_dept.transform(b2["Department"])
                    b2["Edu_enc"]            = le_edu.transform(b2["Education"])
                    b2["Productivity_Index"] = b2["Projects_Handled"] / b2["Experience_Years"].clip(lower=1)
                    b2["Engagement_Score"]   = (b2["Job_Satisfaction"] + b2["Work_Life_Balance"]) / 2
                    b2["Loyalty_Score"]      = b2["Experience_Years"] * (b2["Promotion_Last_5Years"] + 1)
                    b2["Efficiency_Ratio"]   = b2["Projects_Handled"] / (b2["Overtime_Hours"] + 1)
                    b2["Salary_per_Exp"]     = b2["Salary"] / b2["Experience_Years"].clip(lower=1)
                    b2["Training_Attend_Ratio"] = b2["Training_Hours"] * b2["Attendance_Percentage"] / 100
                    b2["Mgr_Sat_Composite"]  = (b2["Manager_Rating"]/5 + b2["Job_Satisfaction"]/5) / 2
                    b2["Perf_Index"]         = ((b2["Manager_Rating"]-1)/4*0.35 +
                                                (b2["Attendance_Percentage"]-60)/40*0.25 +
                                                (b2["Job_Satisfaction"]-1)/4*0.20 +
                                                (b2["Work_Life_Balance"]-1)/4*0.10 +
                                                np.minimum(b2["Projects_Handled"],15)/15*0.06 +
                                                np.minimum(b2["Training_Hours"],40)/40*0.04)

                    Xb = b2[FEATURES].values
                    if USE_SCALED: Xb = scaler.transform(Xb)

                    preds = le_tgt.inverse_transform(model.predict(Xb))
                    bdf["Predicted_Performance"] = preds

                    if hasattr(model, "predict_proba"):
                        proba_b = model.predict_proba(Xb)
                        for i, cls in enumerate(le_tgt.classes_):
                            bdf[f"Prob_{cls}"] = (proba_b[:, i] * 100).round(2)

                st.success(f"✅ Predicted {len(bdf):,} employees!")

                vc2 = bdf["Predicted_Performance"].value_counts()
                sc1,sc2,sc3,sc4 = st.columns(4)
                for col, lbl, ico in zip([sc1,sc2,sc3,sc4],
                                          ["Excellent","Good","Average","Poor"],
                                          ["🌟","👍","⚡","⚠️"]):
                    col.metric(f"{ico} {lbl}", vc2.get(lbl, 0))

                # Summary pie
                fig, ax = plt.subplots(figsize=(4, 3.5))
                vc2_sorted = vc2.reindex(["Excellent","Good","Average","Poor"]).dropna()
                ax.pie(vc2_sorted.values, labels=vc2_sorted.index, autopct="%1.1f%%",
                       colors=PALETTE[:len(vc2_sorted)], startangle=90,
                       wedgeprops={"edgecolor":"white","linewidth":2})
                ax.set_title("Batch Result Distribution", fontweight="bold")
                plt.tight_layout(); st.pyplot(fig); plt.close()

                st.dataframe(bdf, use_container_width=True, height=400)
                st.download_button("📥 Download Predictions CSV",
                                   bdf.to_csv(index=False),
                                   "batch_predictions.csv", "text/csv",
                                   use_container_width=True)

            except Exception as e:
                st.error(f"❌ Error: {e}")
                st.info("Ensure your CSV has all required columns matching the template.")


# ─── Footer ───────────────────────────────────────────────────────────────
st.divider()
col_f1, col_f2, col_f3 = st.columns(3)
col_f1.caption(f"🤖 Model: **{MODEL_NAME}** · Accuracy: **{BEST_ACC*100:.2f}%**")
col_f2.caption(f"📊 Dataset: **{N_SAMPLES:,}** employees · **{N_FEATURES}** features")
col_f3.caption("🐍 Python · scikit-learn · XGBoost · LightGBM · Streamlit")
