import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="Churn Prediction Dashboard",
    page_icon="📊",
    layout="wide"
)

# ─────────────────────────────────────────────
# CUSTOM STYLES
# ─────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid #0f3460;
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        color: white;
    }
    .metric-card .value { font-size: 2rem; font-weight: 700; margin: 0; }
    .metric-card .label { font-size: 0.8rem; color: #a0aec0; margin: 0; }
    .persona-card {
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 0.5rem;
        border-left: 5px solid;
    }
    .section-header {
        font-size: 1.3rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        padding-bottom: 0.3rem;
        border-bottom: 2px solid #e2e8f0;
    }
    .f1-box {
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        margin: 0.3rem;
    }
</style>
""", unsafe_allow_html=True)

st.title("📊 Churn Prediction & Intervention Dashboard")
st.markdown("Upload your **submission CSV** (`user_id` + `churn_status`) to explore results, get per-user interventions, and view business insights.")

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
STATUS_EMOJI = {
    'invol_churn': '🔴',
    'vol_churn':   '🟡',
    'not_churned': '🟢',
}
STATUS_LABEL = {
    'invol_churn': 'Involuntary Churn',
    'vol_churn':   'Voluntary Churn',
    'not_churned': 'Retained',
}
STRATEGIES = {
    'invol_churn': "🔄 PAYMENT RECOVERY: Prompt user to update their credit card. Offer a 7-day grace period with 3 smart retries (off-peak hours).",
    'vol_churn':   "🎁 WIN-BACK: Offer 30% discount + schedule a personal onboarding call. If beginner: trigger guided first session.",
    'not_churned': "✅ HEALTHY: User is retained. Consider for upsell or referral campaign.",
}

# ─────────────────────────────────────────────
# MODEL F1 SCORES (from the presentation)
# ─────────────────────────────────────────────
F1_SCORES = {
    'Macro F1':         0.58,
    'Involuntary Churn F1': 0.62,
    'Voluntary Churn F1':   0.60,
    'Retained F1':          0.52,
}

# ─────────────────────────────────────────────
# STATIC BUSINESS INSIGHT DATA (from PPTX)
# ─────────────────────────────────────────────
CHURN_DISTRIBUTION = pd.DataFrame({
    'Segment':  ['Voluntary Churn', 'Involuntary Churn', 'Retained'],
    'Users':    [28569, 26573, 23425],
    'Pct':      [36.4, 33.8, 29.8],
    'Color':    ['#F59E0B', '#EF4444', '#10B981'],
})

TOP_FEATURES = pd.DataFrame({
    'Feature':  [
        'total_generations',
        'bank_name_churn_rate',
        'country_code_churn_rate',
        'total_credit_spent',
        'avg_credit_per_gen',
        'engagement_decay',
        'frustration_per_gen',
        'unique_gen_types',
    ],
    'Churn Type': ['Both','Involuntary','Involuntary','Both','Voluntary','Voluntary','Voluntary','Both'],
    'Importance': [0.18, 0.15, 0.13, 0.12, 0.10, 0.09, 0.08, 0.07],
})

PAYMENT_METHODS = pd.DataFrame({
    'Method':           ['Link / Digital Wallet', 'Debit Card', 'Credit Card (Standard)', '3DS Card'],
    'Invol Churn Rate': [7, 28, 42, 54],
})

INTERVENTION_TABLE = pd.DataFrame([
    {'Type':'Involuntary','Root Cause':'card_declined at renewal','Intervention':'Smart retry: 3 attempts / 7 days, off-peak','Users':'18,793','Priority':'🔴 CRITICAL'},
    {'Type':'Involuntary','Root Cause':'High-risk bank detected','Intervention':'Pre-renewal verification email 7 days before','Users':'~12,000','Priority':'🟠 HIGH'},
    {'Type':'Involuntary','Root Cause':'No digital wallet on file','Intervention':'Promote Link/PayPal at onboarding & checkout','Users':'All new users','Priority':'🟡 MEDIUM'},
    {'Type':'Voluntary','Root Cause':'Beginner never onboards','Intervention':'Forced guided first session + templates','Users':'7,500+','Priority':'🔴 CRITICAL'},
    {'Type':'Voluntary','Root Cause':'Frustration: Hard to prompt','Intervention':'In-app success chat trigger within 24hrs','Users':'~4,000','Priority':'🟠 HIGH'},
    {'Type':'Voluntary','Root Cause':'Power user hits credit ceiling','Intervention':'Usage dashboard + upgrade prompt at 80% limit','Users':'~3,500','Priority':'🟠 HIGH'},
    {'Type':'Voluntary','Root Cause':'Inactive 14+ days','Intervention':'Win-back email: 30% discount + feature showcase','Users':'15,561','Priority':'🟡 MEDIUM'},
    {'Type':'Both','Root Cause':'Declining usage + bad card','Intervention':'Combined payment fix + re-engagement flow','Users':'~5,000','Priority':'🔴 CRITICAL'},
])

PERSONAS = [
    {
        'name': '😕 The Lost Beginner',
        'type': 'Voluntary',
        'color': '#F59E0B',
        'border': '#F59E0B',
        'desc': '0–5 generations · No experience set · Churns in Week 1–2',
        'fix':  'Guided first session with templates — force a successful generation in first 10 mins.',
        'users': '7,500+'
    },
    {
        'name': '😤 The Frustrated Power User',
        'type': 'Voluntary',
        'color': '#F59E0B',
        'border': '#EF4444',
        'desc': '5,000–20,000+ generations · Frustration: "High cost" · Hits credit ceiling',
        'fix':  'Usage dashboard + upgrade path shown at 80% of credit limit.',
        'users': '~3,500'
    },
    {
        'name': '👻 The Ghost Payer',
        'type': 'Involuntary',
        'color': '#EF4444',
        'border': '#EF4444',
        'desc': 'Normal usage · High-risk bank · Card fails silently at renewal',
        'fix':  'Pre-renewal nudge 7 days before + grace period + smart retry logic.',
        'users': '~12,000'
    },
    {
        'name': '😶 The Quiet Disengager',
        'type': 'Both',
        'color': '#6366F1',
        'border': '#6366F1',
        'desc': '14–21 days inactive · No frustration label · Intermediate/advanced user',
        'fix':  '14-day re-engagement email + 30% discount time-sensitive offer.',
        'users': '15,561'
    },
]

# ─────────────────────────────────────────────
# FILE UPLOAD
# ─────────────────────────────────────────────
uploaded = st.file_uploader(
    "📂 Upload your submission CSV",
    type="csv",
    help="Must have columns: user_id, churn_status (values: invol_churn, vol_churn, not_churned)"
)

user_data_loaded = False
df = None

if uploaded is not None:
    df = pd.read_csv(uploaded)
    if 'churn_status' not in df.columns:
        cols = list(df.columns)
        status_col = st.selectbox("Which column contains the churn status?", cols)
        df = df.rename(columns={status_col: 'churn_status'})
    if 'user_id' not in df.columns:
        cols = [c for c in df.columns if c != 'churn_status']
        id_col = st.selectbox("Which column is the user ID?", cols)
        df = df.rename(columns={id_col: 'user_id'})

    VALID = {'invol_churn', 'vol_churn', 'not_churned'}
    found = set(df['churn_status'].dropna().unique())
    unexpected = found - VALID
    if unexpected:
        st.error(f"❌ Unexpected values in churn_status: `{unexpected}`. Expected: invol_churn, vol_churn, not_churned")
    else:
        df['recommended_action'] = df['churn_status'].map(STRATEGIES)
        df['status_label']       = df['churn_status'].map(STATUS_LABEL)
        user_data_loaded = True
else:
    st.info("👆 Upload your submission CSV to unlock the User Predictions and Export tabs. Business Insights are available without a file.")

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
if user_data_loaded:
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Executive Summary", "🧠 Business Insights", "🔮 User Predictions", "⬇️ Export"])
else:
    tab1, tab2 = st.tabs(["📈 Executive Summary", "🧠 Business Insights"])
    tab3, tab4 = None, None


# ══════════════════════════════════════════════
# TAB 1 — EXECUTIVE SUMMARY
# ══════════════════════════════════════════════
with tab1:
    st.header("Executive Summary")

    # ── Hero KPIs (static from PPTX, or from uploaded file) ──
    if user_data_loaded:
        counts = df['churn_status'].value_counts()
        total  = len(df)
        invol  = counts.get('invol_churn', 0)
        vol    = counts.get('vol_churn', 0)
        retained = counts.get('not_churned', 0)
    else:
        total    = 78567
        invol    = 26573
        vol      = 28569
        retained = 23425
        counts   = pd.Series({'invol_churn': invol, 'vol_churn': vol, 'not_churned': retained})

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("👥 Total Users",         f"{total:,}")
    col2.metric("🔴 Involuntary Churn",   f"{invol:,}",    f"{invol/total*100:.1f}% of users")
    col3.metric("🟡 Voluntary Churn",     f"{vol:,}",      f"{vol/total*100:.1f}% of users")
    col4.metric("🟢 Retained",            f"{retained:,}", f"{retained/total*100:.1f}% of users")

    st.markdown("---")

    # ── Model F1 Scores ──
    st.markdown("### 🎯 Model Performance — F1 Scores (LightGBM)")
    fc1, fc2, fc3, fc4 = st.columns(4)
    f1_cols = [fc1, fc2, fc3, fc4]
    f1_colors = ['#6366F1', '#EF4444', '#F59E0B', '#10B981']
    for i, (label, score) in enumerate(F1_SCORES.items()):
        with f1_cols[i]:
            st.markdown(f"""
            <div style="background:{f1_colors[i]}22; border:2px solid {f1_colors[i]};
                        border-radius:10px; padding:1rem; text-align:center;">
                <div style="font-size:2rem; font-weight:800; color:{f1_colors[i]};">{score:.2f}</div>
                <div style="font-size:0.8rem; color:#4a5568; font-weight:600;">{label}</div>
            </div>
            """, unsafe_allow_html=True)
    st.caption("ℹ️ Macro F1 of 0.58 reflects a hard 3-class problem with near-equal class sizes. The model still correctly flags 34,354 high-risk users.")

    st.markdown("---")

    # ── Churn Distribution bar chart ──
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("### 📊 Churn Distribution")
        if user_data_loaded:
            chart_df = pd.DataFrame({
                'Status': [STATUS_LABEL[k] for k in ['vol_churn','invol_churn','not_churned']],
                'Count':  [counts.get('vol_churn',0), counts.get('invol_churn',0), counts.get('not_churned',0)],
            }).set_index('Status')
        else:
            chart_df = CHURN_DISTRIBUTION[['Segment','Users']].set_index('Segment').rename(columns={'Users':'Count'})
        st.bar_chart(chart_df)

    with col_b:
        st.markdown("### 💡 Key Insights")
        total_at_risk = invol + vol
        pct_at_risk   = total_at_risk / total * 100
        st.error(f"**{pct_at_risk:.1f}%** of users are at risk — {total_at_risk:,} total need intervention.")
        st.warning(
            f"**Involuntary churn** ({invol/total*100:.1f}%) is caused by payment failures, "
            f"NOT product dissatisfaction. Focus on retry logic and wallet promotion first."
        )
        st.info(
            f"**Voluntary churn** ({vol/total*100:.1f}%) is an onboarding & value gap problem. "
            f"Beginners and power users need fundamentally different fixes."
        )
        st.success(
            f"**{retained/total*100:.1f}%** retained users are healthy. "
            f"Target for upsell or referral campaigns."
        )


# ══════════════════════════════════════════════
# TAB 2 — BUSINESS INSIGHTS
# ══════════════════════════════════════════════
with tab2:
    st.header("🧠 Business Insights")
    st.caption("All figures sourced from the HackNU 2026 Retention Architect analysis of 78,567 users.")

    # ── Section 1: Payment Method Risk ──
    st.markdown("### 💳 Involuntary Churn Rate by Payment Method")
    st.markdown("Digital wallets (Link/PayPal) show **~7% involuntary churn** vs 33–54% for traditional cards — a structural fix with massive impact.")

    pm_col1, pm_col2 = st.columns([2, 1])
    with pm_col1:
        pm_chart = PAYMENT_METHODS.set_index('Method')
        st.bar_chart(pm_chart, color='#EF4444')
    with pm_col2:
        st.markdown("""
        <div style="background:#FEF2F2; border-left:4px solid #EF4444; border-radius:8px; padding:1rem; margin-top:1rem;">
            <b>🔑 Key Action</b><br><br>
            Promote Link / PayPal at <b>checkout & onboarding</b>.<br><br>
            Wallet methods <b>bypass card decline scenarios entirely</b> — this is the single highest-leverage structural fix available.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Section 2: Top Predictive Features ──
    st.markdown("### 🔬 Top Predictive Features (LightGBM)")
    feat_col1, feat_col2 = st.columns([2, 1])
    with feat_col1:
        feat_chart = TOP_FEATURES[['Feature', 'Importance']].set_index('Feature')
        st.bar_chart(feat_chart, color='#6366F1')
    with feat_col2:
        color_map = {'Both': '#6366F1', 'Involuntary': '#EF4444', 'Voluntary': '#F59E0B'}
        st.markdown("**Churn Type per Feature:**")
        for _, row in TOP_FEATURES.iterrows():
            c = color_map[row['Churn Type']]
            st.markdown(
                f"<span style='background:{c}22; border:1px solid {c}; border-radius:4px; "
                f"padding:2px 8px; font-size:0.8rem; color:{c}; font-weight:600;'>"
                f"{row['Churn Type']}</span> {row['Feature']}",
                unsafe_allow_html=True
            )

    st.markdown("---")

    # ── Section 3: 4 Churn Personas ──
    st.markdown("### 👤 The 4 Churn Personas — Who Is Leaving?")
    p_cols = st.columns(2)
    for i, p in enumerate(PERSONAS):
        with p_cols[i % 2]:
            st.markdown(f"""
            <div style="border-left:5px solid {p['border']}; background:{p['border']}11;
                        border-radius:8px; padding:1rem; margin-bottom:1rem;">
                <div style="font-size:1.1rem; font-weight:700;">{p['name']}</div>
                <div style="font-size:0.8rem; color:#718096; margin:0.3rem 0;">{p['desc']}</div>
                <div style="margin-top:0.5rem;">
                    <span style="background:{p['border']}22; color:{p['border']}; border-radius:4px;
                                 padding:2px 8px; font-size:0.75rem; font-weight:600;">{p['type']}</span>
                    <span style="font-size:0.75rem; color:#718096; margin-left:0.5rem;">~{p['users']} users</span>
                </div>
                <div style="margin-top:0.7rem; font-size:0.85rem;"><b>Fix:</b> {p['fix']}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Section 4: Intervention Playbook ──
    st.markdown("### 🗺️ Master Intervention Playbook")
    st.dataframe(
        INTERVENTION_TABLE,
        use_container_width=True,
        hide_index=True,
        column_config={
            'Priority': st.column_config.TextColumn('Priority', width='medium'),
            'Users':    st.column_config.TextColumn('Users Affected', width='small'),
        }
    )

    st.markdown("---")

    # ── Section 5: Monday Morning Top 5 ──
    st.markdown("### 🚀 What the Product Team Does Monday Morning")
    st.caption("Ranked by expected retention impact")
    actions = [
        ("🔴", "Deploy 7-day payment grace period with retry logic",
         "Involuntary Fix", "Recovers 30–40% of involuntary churners. Users want to stay — stop ejecting them over a timing failure."),
        ("🔴", "Send pre-renewal payment verification to high-risk banks",
         "Involuntary Fix", "Prevents failures before they occur. Estimated 15–20% reduction in card_declined events."),
        ("🟡", "Redesign beginner first session as a guided, template-driven flow",
         "Voluntary Fix", "Session 1 completion is the #1 predictor of 30-day retention. Fixes the largest single churn cohort."),
        ("🟡", "Build in-app usage dashboard shown 7 days before renewal",
         "Voluntary Fix", "Counteracts 'High cost' frustration by making value tangible exactly when the renewal decision is made."),
        ("🟢", "Trigger success team outreach when frustration field is set",
         "Both", "Turns a passive churn signal into a direct human intervention. Estimated 25–35% save rate for contacted users."),
    ]
    type_colors = {'Involuntary Fix': '#EF4444', 'Voluntary Fix': '#F59E0B', 'Both': '#6366F1'}
    for idx, (emoji, title, atype, desc) in enumerate(actions, 1):
        c = type_colors[atype]
        st.markdown(f"""
        <div style="display:flex; align-items:flex-start; gap:1rem; padding:0.8rem;
                    border:1px solid #e2e8f0; border-radius:10px; margin-bottom:0.5rem; background:white;">
            <div style="font-size:1.5rem; font-weight:800; color:#6366F1; min-width:2rem;">{idx}</div>
            <div>
                <div style="font-weight:700; font-size:1rem;">{emoji} {title}</div>
                <div style="margin:0.2rem 0;">
                    <span style="background:{c}22; color:{c}; border-radius:4px;
                                 padding:1px 8px; font-size:0.75rem; font-weight:600;">{atype}</span>
                </div>
                <div style="font-size:0.85rem; color:#4a5568; margin-top:0.3rem;">{desc}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════
# TAB 3 — USER PREDICTIONS (only if file uploaded)
# ══════════════════════════════════════════════
if tab3 and user_data_loaded:
    with tab3:
        st.header("User-Level Predictions & Interventions")

        col1, col2 = st.columns(2)
        with col1:
            status_filter = st.multiselect(
                "Filter by predicted status",
                options=['invol_churn', 'vol_churn', 'not_churned'],
                default=['invol_churn', 'vol_churn', 'not_churned'],
                format_func=lambda x: f"{STATUS_EMOJI[x]} {STATUS_LABEL[x]}"
            )
        with col2:
            search = st.text_input("🔍 Search by user_id", placeholder="Paste a user_id...")

        filtered = df[df['churn_status'].isin(status_filter)].copy()
        if search.strip():
            filtered = filtered[filtered['user_id'].astype(str).str.contains(search.strip(), case=False, na=False)]

        st.markdown(f"Showing **{len(filtered):,}** users")

        display_df = filtered[['user_id', 'status_label', 'recommended_action']].copy()
        display_df.columns = ['User ID', 'Predicted Status', 'Recommended Action']
        st.dataframe(display_df.reset_index(drop=True), use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════
# TAB 4 — EXPORT (only if file uploaded)
# ══════════════════════════════════════════════
if tab4 and user_data_loaded:
    with tab4:
        st.header("Export Results")

        export_df = df[['user_id', 'churn_status', 'status_label', 'recommended_action']].copy()
        export_df.columns = ['user_id', 'churn_status', 'churn_status_label', 'recommended_action']

        st.dataframe(export_df.head(20), use_container_width=True, hide_index=True)
        st.caption(f"Preview of first 20 rows — full export has {len(export_df):,} users.")

        st.download_button(
            label="⬇️ Download Full Results with Interventions (CSV)",
            data=export_df.to_csv(index=False),
            file_name="churn_predictions_with_interventions.csv",
            mime="text/csv",
        )

        st.markdown("---")
        st.markdown("#### 📋 Original Submission File")
        st.download_button(
            label="⬇️ Download Original Submission CSV",
            data=df[['user_id', 'churn_status']].to_csv(index=False),
            file_name="submission_.csv",
            mime="text/csv",
        )
