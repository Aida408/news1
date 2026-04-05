import streamlit as st
import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Churn Prediction App",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Churn Prediction & Intervention App")
st.markdown("Upload a single cleaned churn CSV, train the model, and get intervention strategies for each user.")

# ─────────────────────────────────────────────
# HELPER: FEATURE ENGINEERING
# ─────────────────────────────────────────────
def engineer_features(df):
    df = df.copy()

    # Recency: days since subscription start
    if 'subscription_start_date' in df.columns:
        df['subscription_start_date'] = pd.to_datetime(df['subscription_start_date'], errors='coerce', utc=True)
        ref_date = pd.Timestamp.now(tz='UTC')
        df['recency_days'] = (ref_date - df['subscription_start_date']).dt.days.fillna(0)
    else:
        df['recency_days'] = 0

    # Payment failure flag
    if 'failure_code' in df.columns:
        df['has_payment_failure'] = df['failure_code'].notna().astype(int)
    else:
        df['has_payment_failure'] = 0

    # High risk card (prepaid or virtual)
    risk_flags = []
    if 'is_prepaid' in df.columns:
        risk_flags.append(df['is_prepaid'].fillna(False).astype(bool))
    if 'is_virtual' in df.columns:
        risk_flags.append(df['is_virtual'].fillna(False).astype(bool))
    if risk_flags:
        df['is_high_risk_card'] = pd.concat(risk_flags, axis=1).any(axis=1).astype(int)
    else:
        df['is_high_risk_card'] = 0

    # Frustration score
    if 'frustration' in df.columns:
        frustration_map = {'low': 1, 'medium': 2, 'high': 3, 'very high': 4}
        df['frustration_score'] = df['frustration'].str.lower().map(frustration_map).fillna(0)
    else:
        df['frustration_score'] = 0

    # Total credit spent
    if 'purchase_amount_dollars' in df.columns:
        df['total_credit_spent'] = df['purchase_amount_dollars'].fillna(0)
    elif 'amount_in_usd' in df.columns:
        df['total_credit_spent'] = df['amount_in_usd'].fillna(0)
    else:
        df['total_credit_spent'] = 0

    # Target encode country_code churn rate
    if 'country_code' in df.columns and 'churn_status' in df.columns:
        cc_map = df.groupby('country_code')['churn_status'].apply(
            lambda x: (x != 'not_churned').mean()
        )
        df['country_code_churn_rate'] = df['country_code'].map(cc_map).fillna(0.5)
    else:
        df['country_code_churn_rate'] = 0.5

    # Target encode bank_name churn rate
    if 'bank_name' in df.columns and 'churn_status' in df.columns:
        bank_map = df.groupby('bank_name')['churn_status'].apply(
            lambda x: (x != 'not_churned').mean()
        )
        df['bank_name_churn_rate'] = df['bank_name'].map(bank_map).fillna(0.5)
    else:
        df['bank_name_churn_rate'] = 0.5

    # Subscription plan encoded
    if 'subscription_plan' in df.columns:
        plan_map = {'Higgsfield Basic': 0, 'Higgsfield Ultimate': 1}
        df['plan_encoded'] = df['subscription_plan'].map(plan_map).fillna(0)
    else:
        df['plan_encoded'] = 0

    # Source encoded
    if 'source' in df.columns:
        sources = ['instagram', 'youtube', 'google', 'tiktok', 'other']
        df['source_encoded'] = pd.Categorical(df['source'].str.lower(), categories=sources).codes
        df['source_encoded'] = df['source_encoded'].replace(-1, len(sources) - 1)
    else:
        df['source_encoded'] = 0

    # total_generations placeholder
    if 'total_generations' not in df.columns:
        df['total_generations'] = 0

    # avg_credit_per_gen
    df['avg_credit_per_gen'] = np.where(
        df['total_generations'] > 0,
        df['total_credit_spent'] / df['total_generations'],
        0
    )

    return df


FEATURE_COLS = [
    'recency_days',
    'has_payment_failure',
    'is_high_risk_card',
    'frustration_score',
    'total_credit_spent',
    'country_code_churn_rate',
    'bank_name_churn_rate',
    'plan_encoded',
    'source_encoded',
    'total_generations',
    'avg_credit_per_gen',
]

VALID_STATUSES = {'invol_churn', 'vol_churn', 'not_churned'}

# ─────────────────────────────────────────────
# HELPER: INTERVENTION STRATEGY
# ─────────────────────────────────────────────
def get_strategy(row):
    status = row['predicted_status']
    if status == 'invol_churn':
        if row['has_payment_failure'] == 1:
            if row['is_high_risk_card'] == 1:
                return "🔄 PAYMENT RECOVERY: Prompt to update to credit card. 7-day grace."
            else:
                return "🔄 RETRY LOGIC: Smart payment retry (3 attempts/7 days). Send reminder."
        else:
            return "⚠️ PRE-EMPTIVE: High-risk card detected. Send verification email."
    elif status == 'vol_churn':
        if row['frustration_score'] >= 3:
            if row['recency_days'] >= 14:
                return "🎁 WIN-BACK: 30% discount + personal onboarding call."
            else:
                return "💬 SUPPORT OUTREACH: Trigger in-app chat from success team."
        elif row['recency_days'] >= 21:
            return "📧 RE-ENGAGEMENT: 'We miss you' email + new feature highlights."
        else:
            return "🎯 VALUE DEMO: Show personalized usage stats and ROI in-app."
    else:
        if row['retained_confidence'] < 0.6:
            return "👀 MONITOR: Low confidence. Add to weekly watch list."
        else:
            return "✅ HEALTHY: Consider for upsell campaign."


# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────
if 'model' not in st.session_state:
    st.session_state.model = None
if 'le' not in st.session_state:
    st.session_state.le = None
if 'df_model' not in st.session_state:
    st.session_state.df_model = None


# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📁 Upload & Train", "🔮 Predictions", "📈 Executive Summary"])


# ══════════════════════════════════════════════
# TAB 1 — UPLOAD & TRAIN
# ══════════════════════════════════════════════
with tab1:
    st.header("Step 1 — Upload your CSV file")

    main_file = st.file_uploader(
        "📂 Upload your churn dataset (CSV)",
        type="csv",
        key="main",
        help="Must contain a 'churn_status' column with values: invol_churn, vol_churn, not_churned"
    )

    gen_file = st.file_uploader(
        "⚙️ Generations CSV (optional — for total_generations feature)",
        type="csv",
        key="gen"
    )

    # ── Column mapping (shown only after file is uploaded) ──────────────
    status_col = 'churn_status'   # default
    if main_file is not None:
        preview = pd.read_csv(main_file, nrows=5)
        main_file.seek(0)          # reset for later full read

        st.markdown("#### 👀 File preview (first 5 rows)")
        st.dataframe(preview, use_container_width=True)

        cols = list(preview.columns)

        if 'churn_status' not in cols:
            st.warning("No `churn_status` column detected. Please map one below.")
            status_col = st.selectbox(
                "Which column contains the churn status?",
                options=cols,
                key="status_col_select"
            )
            st.markdown(
                f"ℹ️ The selected column must contain exactly these values: "
                f"`invol_churn`, `vol_churn`, `not_churned`"
            )
        else:
            st.success("✅ `churn_status` column detected automatically.")

    st.markdown("---")
    st.header("Step 2 — Train the Model")

    if st.button("🚀 Train Model", disabled=(main_file is None)):
        with st.spinner("Loading and engineering features..."):
            df = pd.read_csv(main_file)

            # Rename mapped column to standard name if needed
            if status_col != 'churn_status' and status_col in df.columns:
                df = df.rename(columns={status_col: 'churn_status'})

            # Validate churn_status values
            found_statuses = set(df['churn_status'].dropna().unique())
            unexpected = found_statuses - VALID_STATUSES
            missing = VALID_STATUSES - found_statuses

            if unexpected:
                st.error(
                    f"❌ Unexpected values in `churn_status`: `{unexpected}`. "
                    f"Expected only: `invol_churn`, `vol_churn`, `not_churned`."
                )
                st.stop()

            if missing:
                st.warning(f"⚠️ These churn classes are absent in your data: `{missing}`. "
                           f"The model will only learn from classes present.")

            st.success(f"✅ Loaded {len(df):,} rows — "
                       f"{df['churn_status'].value_counts().to_dict()}")

            # Merge generations if provided
            if gen_file is not None:
                generations = pd.read_csv(gen_file)
                if 'user_id' in generations.columns:
                    gen_agg = generations.groupby('user_id').agg(
                        total_generations=('generation_id', 'count'),
                        total_credit_spent_gen=('credit_cost', 'sum')
                    ).reset_index()
                    df = df.merge(gen_agg, on='user_id', how='left')
                    df['total_generations'] = df['total_generations'].fillna(0)
                    st.success("✅ Merged generations data")

            df = engineer_features(df)

        with st.spinner("Training Random Forest model..."):
            le = LabelEncoder()
            df['target'] = le.fit_transform(df['churn_status'])

            X = df[FEATURE_COLS].fillna(0)
            y = df['target']

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            model.fit(X_train, y_train)

            pickle.dump(model, open("model.pkl", "wb"))
            pickle.dump(le, open("label_encoder.pkl", "wb"))

            st.session_state.model = model
            st.session_state.le = le
            st.session_state.df_model = df

        y_pred = model.predict(X_test)
        report = classification_report(
            y_test, y_pred, target_names=le.classes_, output_dict=True
        )
        report_df = pd.DataFrame(report).transpose().round(3)

        st.success("🎉 Model trained and saved as model.pkl + label_encoder.pkl!")
        st.markdown("### 📊 Model Performance")
        st.dataframe(report_df, use_container_width=True)

        importances = pd.DataFrame({
            'feature': FEATURE_COLS,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)

        st.markdown("### 🔑 Top Feature Importances")
        st.bar_chart(importances.set_index('feature')['importance'])

    elif main_file is None:
        st.info("👆 Please upload a CSV file to enable training.")


# ══════════════════════════════════════════════
# TAB 2 — PREDICTIONS
# ══════════════════════════════════════════════
with tab2:
    st.header("Churn Predictions & Intervention Strategies")

    model    = st.session_state.model
    le       = st.session_state.le
    df_model = st.session_state.df_model

    if model is None:
        st.info("No trained model in session. You can load an existing model.pkl below.")
        col_a, col_b = st.columns(2)
        with col_a:
            loaded_model_file = st.file_uploader("Load model.pkl", type="pkl", key="load_model")
        with col_b:
            loaded_le_file = st.file_uploader("Load label_encoder.pkl", type="pkl", key="load_le")

        if loaded_model_file and loaded_le_file:
            st.session_state.model = pickle.load(loaded_model_file)
            st.session_state.le    = pickle.load(loaded_le_file)
            st.success("✅ Model loaded!")
            st.rerun()
    else:
        st.success("✅ Model is ready!")

        if df_model is not None:
            X_all      = df_model[FEATURE_COLS].fillna(0)
            y_proba    = model.predict_proba(X_all)
            y_pred_all = model.predict(X_all)

            classes = list(le.classes_)
            inv_idx = classes.index('invol_churn') if 'invol_churn' in classes else 0
            vol_idx = classes.index('vol_churn')   if 'vol_churn'   in classes else 1
            nc_idx  = classes.index('not_churned') if 'not_churned' in classes else 2

            interventions = pd.DataFrame({
                'user_id':             df_model['user_id'] if 'user_id' in df_model.columns else range(len(df_model)),
                'predicted_status':    le.inverse_transform(y_pred_all),
                'involuntary_risk':    y_proba[:, inv_idx],
                'voluntary_risk':      y_proba[:, vol_idx],
                'retained_confidence': y_proba[:, nc_idx],
                'recency_days':        df_model['recency_days'].values,
                'has_payment_failure': df_model['has_payment_failure'].values,
                'frustration_score':   df_model['frustration_score'].values,
                'is_high_risk_card':   df_model['is_high_risk_card'].values,
                'total_credit_spent':  df_model['total_credit_spent'].values,
            })

            interventions['recommended_action'] = interventions.apply(get_strategy, axis=1)

            max_credit = interventions['total_credit_spent'].max()
            interventions['priority_score'] = (
                interventions['involuntary_risk'] * 40 +
                interventions['voluntary_risk']   * 35 +
                (interventions['total_credit_spent'] / (max_credit if max_credit > 0 else 1)) * 25
            )
            interventions = interventions.sort_values('priority_score', ascending=False)

            st.markdown("### 🔍 Filter Results")
            col1, col2 = st.columns(2)
            with col1:
                status_filter = st.multiselect(
                    "Predicted Status",
                    options=['invol_churn', 'vol_churn', 'not_churned'],
                    default=['invol_churn', 'vol_churn', 'not_churned']
                )
            with col2:
                top_n = st.slider("Show top N users by priority", 10, 500, 50)

            filtered = interventions[
                interventions['predicted_status'].isin(status_filter)
            ].head(top_n)

            st.markdown(f"### 📋 Top {top_n} Priority Users")
            st.dataframe(
                filtered[[
                    'user_id', 'predicted_status', 'priority_score',
                    'involuntary_risk', 'voluntary_risk', 'retained_confidence',
                    'recommended_action'
                ]].round(3),
                use_container_width=True
            )

            csv = filtered.to_csv(index=False)
            st.download_button(
                label="⬇️ Download Results as CSV",
                data=csv,
                file_name="churn_predictions.csv",
                mime="text/csv"
            )

            with open("model.pkl", "rb") as f:
                st.download_button(
                    label="⬇️ Download model.pkl",
                    data=f,
                    file_name="model.pkl",
                    mime="application/octet-stream"
                )
        else:
            st.warning("Please go to the Upload & Train tab first to load your data.")


# ══════════════════════════════════════════════
# TAB 3 — EXECUTIVE SUMMARY
# ══════════════════════════════════════════════
with tab3:
    st.header("📈 Executive Summary")

    if st.session_state.df_model is None or st.session_state.model is None:
        st.info("Train the model first in the Upload & Train tab.")
    else:
        df_model = st.session_state.df_model
        model    = st.session_state.model
        le       = st.session_state.le

        X_all      = df_model[FEATURE_COLS].fillna(0)
        y_proba    = model.predict_proba(X_all)
        y_pred_all = model.predict(X_all)
        classes    = list(le.classes_)

        inv_idx = classes.index('invol_churn') if 'invol_churn' in classes else 0
        vol_idx = classes.index('vol_churn')   if 'vol_churn'   in classes else 1

        col1, col2, col3, col4 = st.columns(4)
        total_users   = len(df_model)
        high_inv_risk = (y_proba[:, inv_idx] > 0.5).sum()
        high_vol_risk = (y_proba[:, vol_idx] > 0.5).sum()
        at_risk_value = df_model[
            (y_proba[:, inv_idx] > 0.5) | (y_proba[:, vol_idx] > 0.5)
        ]['total_credit_spent'].sum()

        col1.metric("👥 Total Users",           f"{total_users:,}")
        col2.metric("🔴 High Involuntary Risk", f"{high_inv_risk:,}")
        col3.metric("🟡 High Voluntary Risk",   f"{high_vol_risk:,}")
        col4.metric("💰 Value at Risk ($)",     f"${at_risk_value:,.0f}")

        st.markdown("---")

        st.markdown("### 📊 Churn Distribution")
        status_counts = df_model['churn_status'].value_counts().reset_index()
        status_counts.columns = ['Status', 'Count']
        st.bar_chart(status_counts.set_index('Status'))

        st.markdown("### 🔑 Key Predictive Factors")
        importances = pd.DataFrame({
            'feature':    FEATURE_COLS,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False).head(5)
        st.dataframe(importances, use_container_width=True)
