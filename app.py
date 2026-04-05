import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Churn Prediction Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Churn Prediction & Intervention Dashboard")
st.markdown("Upload your **submission CSV** (`user_id` + `churn_status`) to explore results and get per-user intervention strategies.")

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
    'invol_churn': "🔄 PAYMENT RECOVERY: Prompt user to update to a credit card. Offer a 7-day grace period.",
    'vol_churn':   "🎁 WIN-BACK: Offer 30% discount + schedule a personal onboarding call.",
    'not_churned': "✅ HEALTHY: User is retained. Consider for upsell or referral campaign.",
}

# ─────────────────────────────────────────────
# FILE UPLOAD
# ─────────────────────────────────────────────
uploaded = st.file_uploader(
    "📂 Upload your submission CSV",
    type="csv",
    help="Must have columns: user_id, churn_status (values: invol_churn, vol_churn, not_churned)"
)

if uploaded is None:
    st.info("👆 Upload your submission CSV to get started.")
    st.stop()

# ─────────────────────────────────────────────
# LOAD & VALIDATE
# ─────────────────────────────────────────────
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
    st.stop()

df['recommended_action'] = df['churn_status'].map(STRATEGIES)
df['status_label']       = df['churn_status'].map(STATUS_LABEL)

counts = df['churn_status'].value_counts()
total  = len(df)

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📈 Executive Summary", "🔮 User Predictions", "⬇️ Export"])


# ══════════════════════════════════════════════
# TAB 1 — EXECUTIVE SUMMARY
# ══════════════════════════════════════════════
with tab1:
    st.header("Executive Summary")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("👥 Total Users", f"{total:,}")
    col2.metric(
        "🔴 Involuntary Churn",
        f"{counts.get('invol_churn', 0):,}",
        f"{counts.get('invol_churn', 0) / total * 100:.1f}% of users"
    )
    col3.metric(
        "🟡 Voluntary Churn",
        f"{counts.get('vol_churn', 0):,}",
        f"{counts.get('vol_churn', 0) / total * 100:.1f}% of users"
    )
    col4.metric(
        "🟢 Retained",
        f"{counts.get('not_churned', 0):,}",
        f"{counts.get('not_churned', 0) / total * 100:.1f}% of users"
    )

    st.markdown("---")

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("### 📊 Churn Distribution")
        chart_df = pd.DataFrame({
            'Status': ['Involuntary Churn', 'Voluntary Churn', 'Retained'],
            'Count': [
                counts.get('invol_churn', 0),
                counts.get('vol_churn', 0),
                counts.get('not_churned', 0),
            ]
        }).set_index('Status')
        st.bar_chart(chart_df)

    with col_b:
        st.markdown("### 🎯 Intervention Actions Summary")
        action_summary = (
            df.groupby('churn_status')
            .agg(Users=('user_id', 'count'), Action=('recommended_action', 'first'))
            .reset_index()
        )
        action_summary['Segment'] = action_summary['churn_status'].map(STATUS_LABEL)
        st.dataframe(
            action_summary[['Segment', 'Users', 'Action']],
            use_container_width=True,
            hide_index=True
        )

    st.markdown("---")
    st.markdown("### 💡 Key Insights")

    total_at_risk = counts.get('invol_churn', 0) + counts.get('vol_churn', 0)
    pct_at_risk   = total_at_risk / total * 100

    c1, c2, c3 = st.columns(3)
    c1.info(f"**{pct_at_risk:.1f}%** of users are at risk of churning ({total_at_risk:,} total).")
    c2.warning(
        f"**Involuntary churn** is the largest at-risk segment "
        f"({counts.get('invol_churn', 0) / total * 100:.1f}%). "
        f"Focus on payment recovery flows first."
    )
    c3.success(
        f"**{counts.get('not_churned', 0) / total * 100:.1f}%** of users are healthy "
        f"and can be targeted for upsell."
    )


# ══════════════════════════════════════════════
# TAB 2 — USER PREDICTIONS
# ══════════════════════════════════════════════
with tab2:
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
        filtered = filtered[filtered['user_id'].str.contains(search.strip(), case=False, na=False)]

    st.markdown(f"Showing **{len(filtered):,}** users")

    display_df = filtered[['user_id', 'status_label', 'recommended_action']].copy()
    display_df.columns = ['User ID', 'Predicted Status', 'Recommended Action']
    st.dataframe(display_df.reset_index(drop=True), use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════
# TAB 3 — EXPORT
# ══════════════════════════════════════════════
with tab3:
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
