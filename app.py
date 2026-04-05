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
import os

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Churn Prediction App",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Churn Prediction & Intervention App")
st.markdown("Upload your churn dataset, train the model, and get intervention strategies for each user.")

# ─────────────────────────────────────────────
# HELPER: FEATURE ENGINEERING (UNCHANGED)
# ─────────────────────────────────────────────
def engineer_features(df):
    df = df.copy()

    if 'subscription_start_date' in df.columns:
        df['subscription_start_date'] = pd.to_datetime(df['subscription_start_date'], errors='coerce', utc=True)
        ref_date = pd.Timestamp.now(tz='UTC')
        df['recency_days'] = (ref_date - df['subscription_start_date']).dt.days.fillna(0)
    else:
        df['recency_days'] = 0

    if 'failure_code' in df.columns:
        df['has_payment_failure'] = df['failure_code'].notna().astype(int)
    else:
        df['has_payment_failure'] = 0

    risk_flags = []
    if 'is_prepaid' in df.columns:
        risk_flags.append(df['is_prepaid'].fillna(False).astype(bool))
    if 'is_virtual' in df.columns:
        risk_flags.append(df['is_virtual'].fillna(False).astype(bool))
    if risk_flags:
        df['is_high_risk_card'] = pd.concat(risk_flags, axis=1).any(axis=1).astype(int)
    else:
        df['is_high_risk_card'] = 0

    if 'frustration' in df.columns:
        frustration_map = {'low': 1, 'medium': 2, 'high': 3, 'very high': 4}
        df['frustration_score'] = df['frustration'].str.lower().map(frustration_map).fillna(0)
    else:
        df['frustration_score'] = 0

    if 'purchase_amount_dollars' in df.columns:
        df['total_credit_spent'] = df['purchase_amount_dollars'].fillna(0)
    elif 'amount_in_usd' in df.columns:
        df['total_credit_spent'] = df['amount_in_usd'].fillna(0)
    else:
        df['total_credit_spent'] = 0

    if 'country_code' in df.columns and 'churn_status' in df.columns:
        cc_map = df.groupby('country_code')['churn_status'].apply(
            lambda x: (x != 'not_churned').mean()
        )
        df['country_code_churn_rate'] = df['country_code'].map(cc_map).fillna(0.5)
    else:
        df['country_code_churn_rate'] = 0.5

    if 'bank_name' in df.columns and 'churn_status' in df.columns:
        bank_map = df.groupby('bank_name')['churn_status'].apply(
            lambda x: (x != 'not_churned').mean()
        )
        df['bank_name_churn_rate'] = df['bank_name'].map(bank_map).fillna(0.5)
    else:
        df['bank_name_churn_rate'] = 0.5

    if 'subscription_plan' in df.columns:
        plan_map = {'Higgsfield Basic': 0, 'Higgsfield Ultimate': 1}
        df['plan_encoded'] = df['subscription_plan'].map(plan_map).fillna(0)
    else:
        df['plan_encoded'] = 0

    if 'source' in df.columns:
        sources = ['instagram', 'youtube', 'google', 'tiktok', 'other']
        df['source_encoded'] = pd.Categorical(df['source'].str.lower(), categories=sources).codes
        df['source_encoded'] = df['source_encoded'].replace(-1, len(sources) - 1)
    else:
        df['source_encoded'] = 0

    if 'total_generations' not in df.columns:
        df['total_generations'] = 0

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

# ─────────────────────────────────────────────
# STRATEGY (UNCHANGED)
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
# TAB 1 — SINGLE FILE
# ══════════════════════════════════════════════
with tab1:
    st.header("Upload your dataset")

    data_file = st.file_uploader("📁 Churn Dataset CSV", type="csv")
    gen_file = st.file_uploader("⚙️ Generations CSV (optional)", type="csv")

    st.markdown("---")
    st.header("Train the Model")

    if st.button("🚀 Train Model", disabled=not data_file):

        with st.spinner("Loading data..."):
            df = pd.read_csv(data_file)

            if 'churn_status' not in df.columns:
                st.error("Dataset must contain 'churn_status'")
                st.stop()

            st.success(f"✅ Loaded {len(df):,} rows")

            if gen_file is not None:
                generations = pd.read_csv(gen_file)
                if 'user_id' in generations.columns:
                    gen_agg = generations.groupby('user_id').agg(
                        total_generations=('generation_id', 'count'),
                        total_credit_spent_gen=('credit_cost', 'sum')
                    ).reset_index()
                    df = df.merge(gen_agg, on='user_id', how='left')
                    df['total_generations'] = df['total_generations'].fillna(0)

            df = engineer_features(df)

        with st.spinner("Training model..."):
            le = LabelEncoder()
            df['target'] = le.fit_transform(df['churn_status'])

            X = df[FEATURE_COLS].fillna(0)
            y = df['target']

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            model.fit(X_train, y_train)

            pickle.dump(model, open("model.pkl", "wb"))
            pickle.dump(le, open("label_encoder.pkl", "wb"))

            st.session_state.model = model
            st.session_state.le = le
            st.session_state.df_model = df

        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True)
        st.dataframe(pd.DataFrame(report).transpose().round(3))
