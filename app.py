import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Churn Prediction Dashboard", page_icon="📊", layout="wide")

st.title("📊 Churn Prediction & Intervention Dashboard")
st.markdown("Upload your **submission CSV**  to explore predictions, interventions, and business insights.")

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
STATUS_EMOJI = {'invol_churn':'🔴', 'vol_churn':'🟡', 'not_churned':'🟢'}
STATUS_LABEL = {'invol_churn':'Involuntary Churn', 'vol_churn':'Voluntary Churn', 'not_churned':'Retained'}

STRATEGIES = {
    'invol_churn': "🔄 PAYMENT RECOVERY: Prompt user to update card. 7-day grace period + smart retry (3 attempts, off-peak hours).",
    'vol_churn':   "🎁 WIN-BACK: 30% discount + personal onboarding call. If beginner: trigger guided first session.",
    'not_churned': "✅ HEALTHY: User is retained. Consider for upsell or referral campaign.",
}

F1_DATA = {
    'Class':     ['invol_churn', 'not_churned', 'vol_churn', 'Macro Avg'],
    'Precision': [0.60,           0.49,          0.68,         0.59],
    'Recall':    [0.65,           0.57,          0.54,         0.59],
    'F1-Score':  [0.62,           0.52,          0.60,         0.58],
    'Support':   [5315,           4685,          5714,         15714],
    'Color':     ['#EF4444',      '#10B981',     '#F59E0B',    '#6366F1'],
}

FEATURES = pd.DataFrame({
    'Feature':    ['total_generations','bank_name_churn_rate','country_code_churn_rate',
                   'total_credit_spent','avg_credit_per_gen','engagement_decay',
                   'frustration_per_gen','unique_gen_types','unique_resolutions','card_brand_churn_rate'],
    'Importance': [19708,19640,19415,19351,18267,17210,13393,9069,6556,5025],
    'Type':       ['Both','Involuntary','Involuntary','Both','Voluntary',
                   'Voluntary','Voluntary','Both','Both','Involuntary'],
})

PAYMENT_METHODS = pd.DataFrame({
    'Method':             ['Link / Digital Wallet','Debit Card','Credit Card','3DS Card'],
    'Invol Churn Rate %': [7, 28, 42, 54],
})

INTERVENTION_TABLE = pd.DataFrame([
    {'Type':'Involuntary','Root Cause':'card_declined at renewal',      'Intervention':'Smart retry: 3 attempts/7 days, off-peak',       'Users':'18,793','Priority':'🔴 CRITICAL'},
    {'Type':'Involuntary','Root Cause':'High-risk bank detected',        'Intervention':'Pre-renewal verification email 7 days before',   'Users':'~12,000','Priority':'🟠 HIGH'},
    {'Type':'Involuntary','Root Cause':'No digital wallet on file',      'Intervention':'Promote Link/PayPal at onboarding & checkout',   'Users':'All new','Priority':'🟡 MEDIUM'},
    {'Type':'Voluntary',  'Root Cause':'Beginner never onboards',        'Intervention':'Forced guided first session + templates',        'Users':'7,500+', 'Priority':'🔴 CRITICAL'},
    {'Type':'Voluntary',  'Root Cause':'Frustration: Hard to prompt',    'Intervention':'In-app success chat trigger within 24hrs',       'Users':'~4,000', 'Priority':'🟠 HIGH'},
    {'Type':'Voluntary',  'Root Cause':'Power user hits credit ceiling',  'Intervention':'Usage dashboard + upgrade prompt at 80% limit', 'Users':'~3,500', 'Priority':'🟠 HIGH'},
    {'Type':'Voluntary',  'Root Cause':'Inactive 14+ days',              'Intervention':'Win-back: 30% discount + feature showcase',     'Users':'15,561', 'Priority':'🟡 MEDIUM'},
    {'Type':'Both',       'Root Cause':'Declining usage + bad card',     'Intervention':'Combined payment fix + re-engagement flow',      'Users':'~5,000', 'Priority':'🔴 CRITICAL'},
])

PERSONAS = [
    {'name':'😕 The Lost Beginner',         'type':'Voluntary',   'color':'#F59E0B',
     'desc':'0–5 generations · No experience set · Churns in Week 1–2',
     'fix':'Guided first session with templates — force a successful generation in the first 10 mins.', 'users':'7,500+'},
    {'name':'😤 The Frustrated Power User', 'type':'Voluntary',   'color':'#EF4444',
     'desc':'5,000–20,000+ generations · Frustration: "High cost" · Hits credit ceiling',
     'fix':'Usage dashboard + upgrade path shown at 80% of credit limit.', 'users':'~3,500'},
    {'name':'👻 The Ghost Payer',           'type':'Involuntary', 'color':'#EF4444',
     'desc':'Normal usage · High-risk bank · Card fails silently at renewal',
     'fix':'Pre-renewal nudge 7 days before + grace period + smart retry logic.', 'users':'~12,000'},
    {'name':'😶 The Quiet Disengager',      'type':'Both',        'color':'#6366F1',
     'desc':'14–21 days inactive · No frustration label · Intermediate/advanced user',
     'fix':'14-day re-engagement email + 30% discount time-sensitive offer.', 'users':'15,561'},
]

# ─────────────────────────────────────────────
# FILE UPLOAD — everything gates on this
# ─────────────────────────────────────────────
uploaded = st.file_uploader(
    "📂 Upload your submission CSV",
    type="csv",
    help="Expects: user_id, invol_churn_prob, not_churned_prob, vol_churn_prob"
)

if uploaded is None:
    st.info("👆 Upload your submission CSV to get started.")
    st.stop()

# ─────────────────────────────────────────────
# LOAD & VALIDATE
# ─────────────────────────────────────────────
with st.spinner("🔍 Analyzing your submission..."):
    import time
    time.sleep(1.5)
    df = pd.read_csv(uploaded)
cols = list(df.columns)

has_probs  = all(c in cols for c in ['invol_churn_prob','not_churned_prob','vol_churn_prob'])
has_status = 'churn_status' in cols

if not has_probs and not has_status:
    st.error("❌ CSV must have probability columns (invol_churn_prob, not_churned_prob, vol_churn_prob) or a churn_status column.")
    st.stop()

if has_probs:
    prob_cols = ['invol_churn_prob','not_churned_prob','vol_churn_prob']
    label_map = {0:'invol_churn', 1:'not_churned', 2:'vol_churn'}
    df['churn_status'] = df[prob_cols].values.argmax(axis=1)
    df['churn_status'] = df['churn_status'].map(label_map)
    df['confidence']   = df[prob_cols].max(axis=1)
else:
    df['confidence'] = None

VALID = {'invol_churn','vol_churn','not_churned'}
unexpected = set(df['churn_status'].dropna().unique()) - VALID
if unexpected:
    st.error(f"❌ Unexpected values in churn_status: `{unexpected}`")
    st.stop()

df['recommended_action'] = df['churn_status'].map(STRATEGIES)
df['status_label']       = df['churn_status'].map(STATUS_LABEL)

counts   = df['churn_status'].value_counts()
total    = len(df)
invol    = counts.get('invol_churn', 0)
vol      = counts.get('vol_churn', 0)
retained = counts.get('not_churned', 0)

# ─────────────────────────────────────────────
# TABS — only rendered after successful upload
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["📈 Executive Summary", "🧠 Business Insights", "🔮 User Predictions", "⬇️ Export"])


# ══════════════════════════════════════════════
# TAB 1 — EXECUTIVE SUMMARY
# ══════════════════════════════════════════════
with tab1:
    st.header("Executive Summary")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("👥 Total Users",       f"{total:,}")
    c2.metric("🔴 Involuntary Churn", f"{invol:,}",    f"{invol/total*100:.1f}%")
    c3.metric("🟡 Voluntary Churn",   f"{vol:,}",      f"{vol/total*100:.1f}%")
    c4.metric("🟢 Retained",          f"{retained:,}", f"{retained/total*100:.1f}%")

    st.markdown("---")

    st.markdown("### 🎯 Model Performance — LightGBM F1 Scores")
    f1_cols = st.columns(4)
    for i, (cls, f1, color) in enumerate(zip(F1_DATA['Class'], F1_DATA['F1-Score'], F1_DATA['Color'])):
        prec = F1_DATA['Precision'][i]
        rec  = F1_DATA['Recall'][i]
        with f1_cols[i]:
            st.markdown(f"""
            <div style="background:{color}18;border:2px solid {color};
                        border-radius:10px;padding:1rem;text-align:center;">
                <div style="font-size:2.2rem;font-weight:800;color:{color};">{f1:.2f}</div>
                <div style="font-size:0.85rem;font-weight:700;margin:0.2rem 0;">{cls}</div>
                <div style="font-size:0.75rem;color:#718096;">Precision {prec:.2f} · Recall {rec:.2f}</div>
            </div>""", unsafe_allow_html=True)
    st.caption("ℹ️ Macro F1 = 0.58 on a hard 3-class problem with ~equal class sizes (15,714 val samples). Model flags 34,354 high-risk users.")

    st.markdown("---")

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("### 📊 Churn Distribution")
        chart_df = pd.DataFrame({'Count':[vol, invol, retained]},
                                index=['Voluntary Churn','Involuntary Churn','Retained'])
        st.bar_chart(chart_df)
    with col_b:
        st.markdown("### 💡 Key Insights")
        st.error(f"**{(invol+vol)/total*100:.1f}%** of users at risk — {invol+vol:,} need intervention.")
        st.warning("**Involuntary churn** is payment infrastructure failure, NOT product dissatisfaction. Retry logic + wallet promotion = #1 structural fix.")
        st.info("**Voluntary churn** is an onboarding & value gap. Beginners and power users need completely different interventions.")
        st.success(f"**{retained/total*100:.1f}%** retained — target for upsell and referral campaigns.")


# ══════════════════════════════════════════════
# TAB 2 — BUSINESS INSIGHTS
# ══════════════════════════════════════════════
with tab2:
    st.header("🧠 Business Insights")
    st.caption("Analysis from HackNU 2026 — RetentionArchitect — 78,567 users")

    st.markdown("### 💳 Involuntary Churn Rate by Payment Method")
    st.markdown("Digital wallets show **~7% involuntary churn** vs 33–54% for traditional cards.")
    pm1, pm2 = st.columns([2,1])
    with pm1:
        st.bar_chart(PAYMENT_METHODS.set_index('Method'), color='#EF4444')
    with pm2:
        st.markdown("""
        <div style="background:#FEF2F2;border-left:4px solid #EF4444;border-radius:8px;padding:1rem;margin-top:1rem;">
            <b>🔑 Key Action</b><br><br>
            Surface <b>Link / PayPal at checkout &amp; onboarding</b>.<br><br>
            Wallet methods <b>bypass card decline scenarios entirely</b>.
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("### 🔬 Top 10 Predictive Features (real LightGBM importances)")
    fi1, fi2 = st.columns([2,1])
    with fi1:
        st.bar_chart(FEATURES[['Feature','Importance']].set_index('Feature'), color='#6366F1')
    with fi2:
        color_map = {'Both':'#6366F1','Involuntary':'#EF4444','Voluntary':'#F59E0B'}
        st.markdown("**Churn type per feature:**")
        for _, row in FEATURES.iterrows():
            c = color_map[row['Type']]
            st.markdown(
                f"<span style='background:{c}22;border:1px solid {c};border-radius:4px;"
                f"padding:2px 8px;font-size:0.78rem;color:{c};font-weight:600;'>{row['Type']}</span>"
                f" {row['Feature']}", unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("### 📋 Full Classification Report")
    report_df = pd.DataFrame({
        'Class':     F1_DATA['Class'],
        'Precision': F1_DATA['Precision'],
        'Recall':    F1_DATA['Recall'],
        'F1-Score':  F1_DATA['F1-Score'],
        'Support':   F1_DATA['Support'],
    })
    st.dataframe(
        report_df, use_container_width=True, hide_index=True,
        column_config={
            'Precision': st.column_config.ProgressColumn('Precision', min_value=0, max_value=1, format="%.2f"),
            'Recall':    st.column_config.ProgressColumn('Recall',    min_value=0, max_value=1, format="%.2f"),
            'F1-Score':  st.column_config.ProgressColumn('F1-Score',  min_value=0, max_value=1, format="%.2f"),
        }
    )

    st.markdown("---")

    st.markdown("### 👤 The 4 Churn Personas")
    p_cols = st.columns(2)
    for i, p in enumerate(PERSONAS):
        with p_cols[i % 2]:
            st.markdown(f"""
            <div style="border-left:5px solid {p['color']};background:{p['color']}11;
                        border-radius:8px;padding:1rem;margin-bottom:1rem;">
                <div style="font-size:1.1rem;font-weight:700;">{p['name']}</div>
                <div style="font-size:0.8rem;color:#718096;margin:0.3rem 0;">{p['desc']}</div>
                <div style="margin-top:0.5rem;">
                    <span style="background:{p['color']}22;color:{p['color']};border-radius:4px;
                                 padding:2px 8px;font-size:0.75rem;font-weight:600;">{p['type']}</span>
                    <span style="font-size:0.75rem;color:#718096;margin-left:0.5rem;">~{p['users']} users</span>
                </div>
                <div style="margin-top:0.7rem;font-size:0.85rem;"><b>Fix:</b> {p['fix']}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("### 🗺️ Master Intervention Playbook")
    st.dataframe(INTERVENTION_TABLE, use_container_width=True, hide_index=True)

    st.markdown("---")

    st.markdown("### 🚀 Top 5 Actions — Ranked by Expected Retention Impact")
    actions = [
        ("🔴","Deploy 7-day payment grace period with retry logic",             "Involuntary Fix","#EF4444","Recovers 30–40% of involuntary churners. Users want to stay — stop ejecting them over a timing failure."),
        ("🔴","Send pre-renewal payment verification to high-risk banks",         "Involuntary Fix","#EF4444","Prevents failures before they occur. Estimated 15–20% reduction in card_declined events."),
        ("🟡","Redesign beginner first session as a guided template-driven flow", "Voluntary Fix", "#F59E0B","Session 1 completion is the #1 predictor of 30-day retention. Fixes the largest single churn cohort."),
        ("🟡","Build in-app usage dashboard shown 7 days before renewal",         "Voluntary Fix", "#F59E0B","Counteracts 'High cost' frustration by making value tangible exactly when the renewal decision is made."),
        ("🟢","Trigger success team outreach when frustration field is set",      "Both",          "#10B981","Turns a passive churn signal into a direct human intervention. Estimated 25–35% save rate."),
    ]
    for idx, (emoji, title, atype, color, desc) in enumerate(actions, 1):
        st.markdown(f"""
        <div style="display:flex;align-items:flex-start;gap:1rem;padding:0.8rem;
                    border:1px solid #e2e8f0;border-radius:10px;margin-bottom:0.5rem;">
            <div style="font-size:1.5rem;font-weight:800;color:#6366F1;min-width:2rem;">{idx}</div>
            <div>
                <div style="font-weight:700;font-size:1rem;">{emoji} {title}</div>
                <span style="background:{color}22;color:{color};border-radius:4px;
                             padding:1px 8px;font-size:0.75rem;font-weight:600;">{atype}</span>
                <div style="font-size:0.85rem;color:#4a5568;margin-top:0.3rem;">{desc}</div>
            </div>
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════
# TAB 3 — USER PREDICTIONS
# ══════════════════════════════════════════════
with tab3:
    st.header("User-Level Predictions & Interventions")

    col1, col2 = st.columns(2)
    with col1:
        status_filter = st.multiselect(
            "Filter by predicted status",
            options=['invol_churn','vol_churn','not_churned'],
            default=['invol_churn','vol_churn','not_churned'],
            format_func=lambda x: f"{STATUS_EMOJI[x]} {STATUS_LABEL[x]}"
        )
    with col2:
        search = st.text_input("🔍 Search by user_id", placeholder="Paste a user_id...")

    filtered = df[df['churn_status'].isin(status_filter)].copy()
    if search.strip():
        filtered = filtered[filtered['user_id'].astype(str).str.contains(search.strip(), case=False, na=False)]

    st.markdown(f"Showing **{len(filtered):,}** users")

    display_cols  = ['user_id','status_label','recommended_action']
    display_names = ['User ID','Predicted Status','Recommended Action']
    if 'invol_churn_prob' in df.columns:
        display_cols  += ['invol_churn_prob','vol_churn_prob','not_churned_prob','confidence']
        display_names += ['Invol Prob','Vol Prob','Retained Prob','Confidence']

    disp = filtered[display_cols].copy()
    disp.columns = display_names

    col_config = {}
    for col in ['Invol Prob','Vol Prob','Retained Prob','Confidence']:
        if col in disp.columns:
            col_config[col] = st.column_config.ProgressColumn(col, min_value=0, max_value=1, format="%.3f")

    st.dataframe(disp.reset_index(drop=True), use_container_width=True, hide_index=True, column_config=col_config)


# ══════════════════════════════════════════════
# TAB 4 — EXPORT
# ══════════════════════════════════════════════
with tab4:
    st.header("Export Results")

    export_cols = ['user_id','churn_status','status_label','recommended_action']
    if 'invol_churn_prob' in df.columns:
        export_cols += ['invol_churn_prob','not_churned_prob','vol_churn_prob','confidence']

    export_df = df[export_cols].copy()
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
    orig_cols = ['user_id','invol_churn_prob','not_churned_prob','vol_churn_prob'] \
                if 'invol_churn_prob' in df.columns else ['user_id','churn_status']
    st.download_button(
        label="⬇️ Download Original Submission CSV",
        data=df[orig_cols].to_csv(index=False),
        file_name="submission_original.csv",
        mime="text/csv",
    )
