import streamlit as st
import pandas as pd
import numpy as np
import pickle
import warnings
from datetime import timedelta
warnings.filterwarnings('ignore')

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
from lightgbm import LGBMClassifier, early_stopping, log_evaluation

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Churn Prediction App",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Churn Prediction & Intervention App")
st.markdown(
    "Upload your **churn dataset** (with a `churn_status` column) and the "
    "**generations CSV** to train a LightGBM model and get per-user intervention strategies."
)

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
VALID_STATUSES = {'Involuntary', 'Voluntary', 'Retained'}

FRUSTRATION_MAP = {
    'Null': 0, 'None': 0, None: 0,
    'Other': 1,
    'Limited generations': 2,
    'Hard to prompt': 3,
    'High cost': 4,
}

EXPERIENCE_MAP = {
    'Beginner': 1,
    'Intermediate': 2,
    'Advanced': 3,
    'Expert': 4,
}

NUMERIC_FEATURES = [
    # Generation / RFM
    'total_generations', 'total_failures', 'overall_fail_rate',
    'total_credit_spent', 'avg_credit_per_gen', 'recency_days',
    'gen_frequency', 'gens_last_7d', 'gens_last_30d',
    'fail_rate_7d', 'fail_rate_30d', 'activity_trend_7d',
    'activity_trend_30d', 'failure_spike_7d',
    'recent_consecutive_fails', 'engagement_decay',
    # Payment
    'is_high_risk_card', 'is_debit', 'has_3d_secure', 'weak_security',
    'cvc_failed', 'cvc_unchecked', 'is_cross_border', 'has_payment_failure',
    'is_high_value_purchase',
    # User profile
    'frustration_score', 'experience_level', 'tenure_days',
    'is_new_user', 'is_veteran', 'is_professional',
    # Interaction
    'frustration_per_gen', 'gen_density', 'disengaged_frustrated',
    'active_but_payment_issue', 'tech_frustration', 'engagement_health',
    # Target encodings
    'country_code_churn_rate', 'card_brand_churn_rate',
    'bank_name_churn_rate', 'role_churn_rate',
]

CATEGORICAL_FEATURES = ['experience', 'role']


# ─────────────────────────────────────────────
# FEATURE ENGINEERING — GENERATIONS
# ─────────────────────────────────────────────
def build_gen_features(generations: pd.DataFrame) -> pd.DataFrame:
    generations = generations.copy()
    generations['created_at'] = pd.to_datetime(generations['created_at'], errors='coerce')
    generations['failed_at']  = pd.to_datetime(generations['failed_at'],  errors='coerce')

    reference_date = generations['created_at'].max()

    # Basic aggregations
    gen_features = generations.groupby('user_id').agg(
        total_generations   = ('generation_id', 'count'),
        total_failures      = ('failed_at',     lambda x: x.notnull().sum()),
        overall_fail_rate   = ('failed_at',     lambda x: x.notnull().mean()),
        total_credit_spent  = ('credit_cost',   'sum'),
        avg_credit_per_gen  = ('credit_cost',   'mean'),
        first_gen_date      = ('created_at',    'min'),
        last_gen_date       = ('created_at',    'max'),
    ).reset_index()

    # RFM
    gen_features['recency_days']    = (reference_date - gen_features['last_gen_date']).dt.days
    gen_features['active_span_days'] = (gen_features['last_gen_date'] - gen_features['first_gen_date']).dt.days + 1
    gen_features['gen_frequency']   = gen_features['total_generations'] / gen_features['active_span_days'].clip(lower=1)

    # Trends
    cutoff_7d  = reference_date - timedelta(days=7)
    cutoff_30d = reference_date - timedelta(days=30)

    recent_7d = (
        generations[generations['created_at'] >= cutoff_7d]
        .groupby('user_id').agg(
            gens_last_7d  = ('generation_id', 'count'),
            fails_last_7d = ('failed_at',     lambda x: x.notnull().sum()),
            fail_rate_7d  = ('failed_at',     lambda x: x.notnull().mean()),
        ).reset_index()
    )
    recent_30d = (
        generations[generations['created_at'] >= cutoff_30d]
        .groupby('user_id').agg(
            gens_last_30d  = ('generation_id', 'count'),
            fails_last_30d = ('failed_at',     lambda x: x.notnull().sum()),
            fail_rate_30d  = ('failed_at',     lambda x: x.notnull().mean()),
        ).reset_index()
    )

    gen_features = gen_features.merge(recent_7d,  on='user_id', how='left')
    gen_features = gen_features.merge(recent_30d, on='user_id', how='left')

    for col in ['gens_last_7d', 'fails_last_7d', 'fail_rate_7d',
                'gens_last_30d', 'fails_last_30d', 'fail_rate_30d']:
        gen_features[col] = gen_features[col].fillna(0)

    gen_features['activity_trend_7d']  = gen_features['gens_last_7d'] / (gen_features['total_generations'] / 4).clip(lower=0.1)
    gen_features['activity_trend_30d'] = gen_features['gens_last_30d'] / gen_features['total_generations'].clip(lower=1)
    gen_features['failure_spike_7d']   = (gen_features['fail_rate_7d'] > gen_features['overall_fail_rate'] * 1.5).astype(int)

    # Consecutive recent failures (last 5 gens)
    gens_sorted = generations.sort_values(['user_id', 'created_at'])
    recent_consecutive = (
        gens_sorted.groupby('user_id')
        .tail(5)
        .groupby('user_id')['failed_at']
        .apply(lambda x: x.notnull().sum())
        .rename('recent_consecutive_fails')
        .reset_index()
    )
    gen_features = gen_features.merge(recent_consecutive, on='user_id', how='left')
    gen_features['recent_consecutive_fails'] = gen_features['recent_consecutive_fails'].fillna(0)

    # Engagement decay (recency-weighted activity)
    gen_features['engagement_decay'] = gen_features['recency_days'] / (gen_features['total_generations'] + 1)

    return gen_features


# ─────────────────────────────────────────────
# FEATURE ENGINEERING — MAIN DATAFRAME
# ─────────────────────────────────────────────
def build_features(df: pd.DataFrame, gen_features: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Merge generation features
    df = df.merge(gen_features, on='user_id', how='left')
    gen_cols = [c for c in gen_features.columns if c != 'user_id']
    df[gen_cols] = df[gen_cols].fillna(0)

    # ── Payment risk ──────────────────────────────────────────
    df['is_high_risk_card'] = (
        (df.get('is_prepaid', pd.Series(0, index=df.index)) == 1) |
        (df.get('is_virtual', pd.Series(0, index=df.index)) == 1)
    ).astype(int)

    df['is_debit'] = (
        df['card_funding'].astype(str).str.lower() == 'debit'
        if 'card_funding' in df.columns else pd.Series(0, index=df.index)
    ).astype(int)

    df['has_3d_secure'] = (
        (df['is_3d_secure'] == 1).astype(int)
        if 'is_3d_secure' in df.columns else pd.Series(0, index=df.index)
    )

    if 'cvc_check' in df.columns:
        df['cvc_failed']    = (df['cvc_check'] == 'fail').astype(int)
        df['cvc_unchecked'] = (df['cvc_check'] == 'unchecked').astype(int)
        df['weak_security'] = ((df['has_3d_secure'] == 0) & (df['cvc_check'] != 'pass')).astype(int)
    else:
        df['cvc_failed']    = 0
        df['cvc_unchecked'] = 0
        df['weak_security'] = 0

    df['is_cross_border'] = (
        (df['country_code'] != df['bank_country']).astype(int)
        if ('country_code' in df.columns and 'bank_country' in df.columns)
        else pd.Series(0, index=df.index)
    )

    df['has_payment_failure'] = (
        df['failure_code'].notnull().astype(int)
        if 'failure_code' in df.columns else pd.Series(0, index=df.index)
    )

    if 'purchase_amount_dollars' in df.columns:
        q75 = df['purchase_amount_dollars'].quantile(0.75)
        df['is_high_value_purchase'] = (df['purchase_amount_dollars'] > q75).astype(int)
    else:
        df['is_high_value_purchase'] = 0

    # ── User profile ──────────────────────────────────────────
    df['frustration_score'] = df['frustration'].map(FRUSTRATION_MAP).fillna(0) if 'frustration' in df.columns else 0
    df['experience_level']  = df['experience'].map(EXPERIENCE_MAP).fillna(2)  if 'experience'  in df.columns else 2

    df['subscription_start_date'] = pd.to_datetime(df.get('subscription_start_date'), errors='coerce')
    df['transaction_time']        = pd.to_datetime(df.get('transaction_time'),        errors='coerce')
    df['tenure_days'] = (df['transaction_time'] - df['subscription_start_date']).dt.days
    df['tenure_days'] = df['tenure_days'].fillna(df['tenure_days'].median()).clip(lower=0)

    df['is_new_user'] = (df['tenure_days'] <= 30).astype(int)
    df['is_veteran']  = (df['tenure_days'] >= 180).astype(int)

    if 'role' in df.columns:
        pro_roles = ['designer', 'developer', 'marketer', 'creator', 'professional']
        df['is_professional'] = df['role'].astype(str).str.lower().str.contains('|'.join(pro_roles), na=False).astype(int)
    else:
        df['is_professional'] = 0

    # ── Target encoding ───────────────────────────────────────
    le = LabelEncoder()
    df['target'] = le.fit_transform(df['churn_status'])

    for col in ['country_code', 'card_brand', 'bank_name', 'role']:
        if col in df.columns:
            mapping = df.groupby(col)['target'].mean()
            df[f'{col}_churn_rate'] = df[col].map(mapping).fillna(0)

    # ── Interaction features ──────────────────────────────────
    df['frustration_per_gen'] = df['frustration_score'] / (df['total_generations'] + 1)
    df['gen_density']         = df['total_generations'] / (df['tenure_days'] + 1)

    df['disengaged_frustrated']    = ((df['frustration_score'] >= 3) & (df['recency_days'] >= 14)).astype(int)
    df['active_but_payment_issue'] = ((df['has_payment_failure'] == 1) & (df['recency_days'] <= 7)).astype(int)

    df['tech_frustration'] = (
        df['recent_consecutive_fails'] * 2 +
        df['fail_rate_7d'] * 10 +
        df['cvc_failed'] * 3
    )

    df['engagement_health'] = (
        (1 - df['overall_fail_rate']) * 25 +
        df['activity_trend_7d'].clip(0, 2) * 25 +
        (1 / (df['recency_days'] + 1)) * 25
    )

    return df, le


# ─────────────────────────────────────────────
# INTERVENTION STRATEGY
# ─────────────────────────────────────────────
def get_strategy(row):
    if row['predicted_status'] == 'Involuntary':
        if row['has_payment_failure'] == 1:
            return ("🔄 PAYMENT RECOVERY: Prompt to update to credit card. 7-day grace."
                    if row['is_high_risk_card'] == 1
                    else "🔄 RETRY LOGIC: Smart payment retry (3 attempts/7 days). Send reminder.")
        return "⚠️ PRE-EMPTIVE: High-risk card detected. Send verification email."

    elif row['predicted_status'] == 'Voluntary':
        if row['frustration_score'] >= 3:
            return ("🎁 WIN-BACK: 30% discount + personal onboarding call."
                    if row['recency_days'] >= 14
                    else "💬 SUPPORT OUTREACH: Trigger in-app chat from success team.")
        return ("📧 RE-ENGAGEMENT: 'We miss you' email + new feature highlights."
                if row['recency_days'] >= 21
                else "🎯 VALUE DEMO: Show personalized usage stats and ROI in-app.")

    else:  # Retained
        return ("👀 MONITOR: Low confidence. Add to weekly watch list."
                if row['retained_confidence'] < 0.6
                else "✅ HEALTHY: Consider for upsell campaign.")


# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────
for key in ['model', 'le', 'df_model', 'feature_names', 'importances']:
    if key not in st.session_state:
        st.session_state[key] = None


# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📁 Upload & Train", "🔮 Predictions", "📈 Executive Summary"])


# ══════════════════════════════════════════════
# TAB 1 — UPLOAD & TRAIN
# ══════════════════════════════════════════════
with tab1:
    st.header("Step 1 — Upload your data")

    col_l, col_r = st.columns(2)
    with col_l:
        main_file = st.file_uploader(
            "📂 Churn dataset CSV",
            type="csv",
            help="Must contain a column with values: Involuntary, Voluntary, Retained"
        )
    with col_r:
        gen_file = st.file_uploader(
            "⚙️ Generations CSV",
            type="csv",
            help="Needs columns: user_id, generation_id, credit_cost, created_at, failed_at"
        )

    status_col = 'churn_status'

    if main_file:
        preview = pd.read_csv(main_file, nrows=5)
        main_file.seek(0)
        st.markdown("#### 👀 Churn file preview")
        st.dataframe(preview, use_container_width=True)

        cols = list(preview.columns)
        if 'churn_status' not in cols:
            st.warning("No `churn_status` column found — please map it below.")
            status_col = st.selectbox("Which column contains churn status?", cols)
            st.info("Expected values: `Involuntary`, `Voluntary`, `Retained`")
        else:
            st.success("✅ `churn_status` column auto-detected.")

    st.markdown("---")
    st.header("Step 2 — Train the Model")

    can_train = main_file is not None and gen_file is not None
    if not can_train:
        st.info("👆 Upload both CSV files to enable training.")

    if st.button("🚀 Train LightGBM Model", disabled=not can_train):

        # ── Load data ────────────────────────────────────────
        with st.spinner("Loading data..."):
            df = pd.read_csv(main_file)
            generations = pd.read_csv(gen_file)

            if status_col != 'churn_status' and status_col in df.columns:
                df = df.rename(columns={status_col: 'churn_status'})

            found = set(df['churn_status'].dropna().unique())
            unexpected = found - VALID_STATUSES
            if unexpected:
                st.error(f"❌ Unexpected churn_status values: `{unexpected}`. Expected: Involuntary, Voluntary, Retained.")
                st.stop()

            st.success(f"✅ Loaded {len(df):,} users — {df['churn_status'].value_counts().to_dict()}")
            st.success(f"✅ Loaded {len(generations):,} generation events")

        # ── Feature engineering ──────────────────────────────
        with st.spinner("Engineering features (RFM, trends, payment risk)..."):
            gen_features = build_gen_features(generations)
            df_model, le = build_features(df, gen_features)
            st.success(f"✅ Feature matrix ready: {df_model.shape[0]:,} rows")

        # ── Prepare X / y ────────────────────────────────────
        available_numeric     = [c for c in NUMERIC_FEATURES     if c in df_model.columns]
        available_categorical = [c for c in CATEGORICAL_FEATURES if c in df_model.columns]
        feature_names         = available_numeric + available_categorical

        X = df_model[feature_names].copy()
        y = df_model['target']

        X[available_numeric] = X[available_numeric].fillna(0)
        for col in available_categorical:
            X[col] = X[col].astype('category')

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )

        st.info(f"Using **{len(feature_names)}** features "
                f"({len(available_numeric)} numeric + {len(available_categorical)} categorical)")

        # ── Train ────────────────────────────────────────────
        with st.spinner("Training LightGBM with early stopping…"):
            model = LGBMClassifier(
                n_estimators    = 2000,
                learning_rate   = 0.02,
                num_leaves      = 31,
                min_child_samples = 50,
                max_depth       = 6,
                class_weight    = 'balanced',
                colsample_bytree = 0.8,
                subsample       = 0.8,
                reg_alpha       = 0.1,
                reg_lambda      = 0.1,
                force_col_wise  = True,
                random_state    = 42,
                verbose         = -1,
            )
            model.fit(
                X_train, y_train,
                eval_set        = [(X_val, y_val)],
                eval_metric     = 'multi_logloss',
                callbacks       = [
                    early_stopping(stopping_rounds=100),
                    log_evaluation(period=200),
                ],
            )

        # ── Evaluate ─────────────────────────────────────────
        y_pred     = model.predict(X_val)
        macro_f1   = f1_score(y_val, y_pred, average='macro')
        report     = classification_report(y_val, y_pred, target_names=le.classes_, output_dict=True)
        report_df  = pd.DataFrame(report).transpose().round(3)

        importances = pd.DataFrame({
            'feature':    feature_names,
            'importance': model.feature_importances_,
        }).sort_values('importance', ascending=False)

        # Save to session
        pickle.dump(model, open("model.pkl", "wb"))
        pickle.dump(le,    open("label_encoder.pkl", "wb"))

        st.session_state.model        = model
        st.session_state.le           = le
        st.session_state.df_model     = df_model
        st.session_state.feature_names = feature_names
        st.session_state.importances  = importances

        # ── Display results ──────────────────────────────────
        st.success(f"🎉 Model trained! **Macro F1 = {macro_f1:.4f}**")

        st.markdown("### 📊 Model Performance")
        st.dataframe(report_df, use_container_width=True)

        st.markdown("### 🔑 Top 15 Feature Importances")
        st.bar_chart(importances.set_index('feature')['importance'].head(15))


# ══════════════════════════════════════════════
# TAB 2 — PREDICTIONS
# ══════════════════════════════════════════════
with tab2:
    st.header("Churn Predictions & Intervention Strategies")

    model         = st.session_state.model
    le            = st.session_state.le
    df_model      = st.session_state.df_model
    feature_names = st.session_state.feature_names

    if model is None:
        st.info("No trained model in session. Load existing pkl files below.")
        col_a, col_b = st.columns(2)
        with col_a:
            loaded_model = st.file_uploader("Load model.pkl", type="pkl", key="load_model")
        with col_b:
            loaded_le    = st.file_uploader("Load label_encoder.pkl", type="pkl", key="load_le")

        if loaded_model and loaded_le:
            st.session_state.model = pickle.load(loaded_model)
            st.session_state.le    = pickle.load(loaded_le)
            st.success("✅ Model loaded! Re-run from Upload & Train to restore predictions.")
            st.rerun()

    else:
        st.success("✅ Model is ready!")

        if df_model is not None:
            X_all = df_model[feature_names].copy()
            for col in [c for c in CATEGORICAL_FEATURES if c in X_all.columns]:
                X_all[col] = X_all[col].astype('category')
            X_all[[c for c in feature_names if c not in CATEGORICAL_FEATURES]] = \
                X_all[[c for c in feature_names if c not in CATEGORICAL_FEATURES]].fillna(0)

            y_proba    = model.predict_proba(X_all)
            y_pred_all = model.predict(X_all)

            classes = list(le.classes_)
            inv_idx = classes.index('Involuntary') if 'Involuntary' in classes else 0
            vol_idx = classes.index('Voluntary')   if 'Voluntary'   in classes else 1
            nc_idx  = classes.index('Retained')    if 'Retained'    in classes else 2

            interventions = pd.DataFrame({
                'user_id':              df_model['user_id'] if 'user_id' in df_model.columns else range(len(df_model)),
                'predicted_status':     le.inverse_transform(y_pred_all),
                'involuntary_risk':     y_proba[:, inv_idx],
                'voluntary_risk':       y_proba[:, vol_idx],
                'retained_confidence':  y_proba[:, nc_idx],
                'recency_days':         df_model['recency_days'].values,
                'has_payment_failure':  df_model['has_payment_failure'].values,
                'frustration_score':    df_model['frustration_score'].values,
                'is_high_risk_card':    df_model['is_high_risk_card'].values,
                'total_credit_spent':   df_model['total_credit_spent'].values,
            })

            interventions['recommended_action'] = interventions.apply(get_strategy, axis=1)

            max_credit = interventions['total_credit_spent'].max()
            interventions['priority_score'] = (
                interventions['involuntary_risk'] * 40 +
                interventions['voluntary_risk']   * 35 +
                (interventions['total_credit_spent'] / (max_credit if max_credit > 0 else 1)) * 25
            )
            interventions = interventions.sort_values('priority_score', ascending=False)

            # Filters
            st.markdown("### 🔍 Filter Results")
            col1, col2 = st.columns(2)
            with col1:
                status_filter = st.multiselect(
                    "Predicted Status",
                    options=['Involuntary', 'Voluntary', 'Retained'],
                    default=['Involuntary', 'Voluntary', 'Retained'],
                )
            with col2:
                top_n = st.slider("Show top N users by priority", 10, 500, 50)

            filtered = interventions[interventions['predicted_status'].isin(status_filter)].head(top_n)

            st.markdown(f"### 📋 Top {top_n} Priority Users")
            st.dataframe(
                filtered[[
                    'user_id', 'predicted_status', 'priority_score',
                    'involuntary_risk', 'voluntary_risk', 'retained_confidence',
                    'recommended_action',
                ]].round(3),
                use_container_width=True,
            )

            st.download_button(
                "⬇️ Download Results as CSV",
                data=filtered.to_csv(index=False),
                file_name="churn_predictions.csv",
                mime="text/csv",
            )
            with open("model.pkl", "rb") as f:
                st.download_button(
                    "⬇️ Download model.pkl",
                    data=f,
                    file_name="model.pkl",
                    mime="application/octet-stream",
                )

        else:
            st.warning("Train the model first in the Upload & Train tab.")


# ══════════════════════════════════════════════
# TAB 3 — EXECUTIVE SUMMARY
# ══════════════════════════════════════════════
with tab3:
    st.header("📈 Executive Summary")

    if st.session_state.df_model is None or st.session_state.model is None:
        st.info("Train the model first in the Upload & Train tab.")
    else:
        df_model      = st.session_state.df_model
        model         = st.session_state.model
        le            = st.session_state.le
        feature_names = st.session_state.feature_names
        importances   = st.session_state.importances

        X_all = df_model[feature_names].copy()
        for col in [c for c in CATEGORICAL_FEATURES if c in X_all.columns]:
            X_all[col] = X_all[col].astype('category')
        X_all[[c for c in feature_names if c not in CATEGORICAL_FEATURES]] = \
            X_all[[c for c in feature_names if c not in CATEGORICAL_FEATURES]].fillna(0)

        y_proba    = model.predict_proba(X_all)
        classes    = list(le.classes_)
        inv_idx    = classes.index('Involuntary') if 'Involuntary' in classes else 0
        vol_idx    = classes.index('Voluntary')   if 'Voluntary'   in classes else 1

        # KPI cards
        col1, col2, col3, col4 = st.columns(4)
        total_users   = len(df_model)
        high_inv_risk = (y_proba[:, inv_idx] > 0.5).sum()
        high_vol_risk = (y_proba[:, vol_idx] > 0.5).sum()
        at_risk_mask  = (y_proba[:, inv_idx] > 0.5) | (y_proba[:, vol_idx] > 0.5)
        at_risk_value = df_model.loc[at_risk_mask, 'total_credit_spent'].sum()

        col1.metric("👥 Total Users",           f"{total_users:,}")
        col2.metric("🔴 High Involuntary Risk", f"{high_inv_risk:,}")
        col3.metric("🟡 High Voluntary Risk",   f"{high_vol_risk:,}")
        col4.metric("💰 Value at Risk ($)",     f"${at_risk_value:,.0f}")

        st.markdown("---")

        st.markdown("### 📊 Churn Distribution")
        status_counts = df_model['churn_status'].value_counts().reset_index()
        status_counts.columns = ['Status', 'Count']
        st.bar_chart(status_counts.set_index('Status'))

        st.markdown("### 🔑 Top 10 Predictive Factors")
        st.bar_chart(importances.set_index('feature')['importance'].head(10))

        st.markdown("### 📋 Top Recommended Actions")
        # Re-build interventions for summary
        y_pred_all = model.predict(X_all)
        nc_idx     = classes.index('Retained') if 'Retained' in classes else 2
        interventions_s = pd.DataFrame({
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
        interventions_s['recommended_action'] = interventions_s.apply(get_strategy, axis=1)

        action_summary = (
            interventions_s['recommended_action']
            .value_counts()
            .reset_index()
        )
        action_summary.columns = ['Action', 'Users Affected']
        st.dataframe(action_summary, use_container_width=True)
