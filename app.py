import streamlit as st
import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Churn Prediction App - HackNU 2026",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Churn Prediction & Intervention System")
st.markdown("### HackNU 2026 - The Retention Architect")
st.markdown("Upload your cleaned dataset, train the model, and get intervention strategies for at-risk users.")

# ─────────────────────────────────────────────
# FEATURE ENGINEERING FUNCTION
# ─────────────────────────────────────────────
def engineer_features(df):
    """Engineer features from raw data"""
    df = df.copy()
    
    # 1. Recency features (if date columns exist)
    if 'subscription_start_date' in df.columns:
        df['subscription_start_date'] = pd.to_datetime(df['subscription_start_date'], errors='coerce')
        ref_date = pd.Timestamp.now()
        df['recency_days'] = (ref_date - df['subscription_start_date']).dt.days.fillna(0)
    else:
        df['recency_days'] = 0
    
    # 2. Payment failure features
    if 'failure_code' in df.columns:
        df['has_payment_failure'] = df['failure_code'].notna().astype(int)
        # Count failures per user (if multiple rows)
        if 'user_id' in df.columns:
            failure_counts = df.groupby('user_id')['has_payment_failure'].sum().reset_index()
            failure_counts.columns = ['user_id', 'payment_failure_count']
            df = df.merge(failure_counts, on='user_id', how='left')
            df['payment_failure_count'] = df['payment_failure_count'].fillna(0)
        else:
            df['payment_failure_count'] = df['has_payment_failure']
    else:
        df['has_payment_failure'] = 0
        df['payment_failure_count'] = 0
    
    # 3. High risk card detection
    risk_flags = []
    if 'is_prepaid' in df.columns:
        risk_flags.append(df['is_prepaid'].fillna(False).astype(bool))
    if 'is_virtual' in df.columns:
        risk_flags.append(df['is_virtual'].fillna(False).astype(bool))
    if risk_flags:
        df['is_high_risk_card'] = pd.concat(risk_flags, axis=1).any(axis=1).astype(int)
    else:
        df['is_high_risk_card'] = 0
    
    # 4. Frustration score
    if 'frustration' in df.columns:
        frustration_map = {'low': 1, 'medium': 2, 'high': 3, 'very high': 4}
        df['frustration_score'] = df['frustration'].str.lower().map(frustration_map).fillna(0)
    else:
        df['frustration_score'] = 0
    
    # 5. Total credit spent
    if 'purchase_amount_dollars' in df.columns:
        df['total_credit_spent'] = df['purchase_amount_dollars'].fillna(0)
    elif 'amount_in_usd' in df.columns:
        df['total_credit_spent'] = df['amount_in_usd'].fillna(0)
    else:
        df['total_credit_spent'] = 0
    
    # 6. Engagement score
    engagement_cols = []
    if 'num_quizzes' in df.columns:
        engagement_cols.append(df['num_quizzes'].fillna(0))
    if 'num_purchases' in df.columns:
        engagement_cols.append(df['num_purchases'].fillna(0))
    if 'num_transactions' in df.columns:
        engagement_cols.append(df['num_transactions'].fillna(0))
    
    if engagement_cols:
        df['engagement_score'] = sum(engagement_cols) / len(engagement_cols)
    else:
        df['engagement_score'] = 0
    
    # 7. Target encode categorical features
    if 'country_code' in df.columns and 'churn_status' in df.columns:
        cc_map = df.groupby('country_code')['churn_status'].apply(
            lambda x: (x != 'not_churned').mean()
        )
        df['country_code_churn_rate'] = df['country_code'].map(cc_map).fillna(0.5)
    else:
        df['country_code_churn_rate'] = 0.5
    
    # 8. Subscription plan encoded
    if 'subscription_plan' in df.columns:
        plan_map = {'Higgsfield Basic': 0, 'Higgsfield Creator': 1, 
                    'Higgsfield Pro': 2, 'Higgsfield Ultimate': 3}
        df['plan_encoded'] = df['subscription_plan'].map(plan_map).fillna(1)
    else:
        df['plan_encoded'] = 0
    
    # 9. Source encoded (if exists)
    if 'source' in df.columns:
        common_sources = ['instagram', 'youtube', 'tiktok', 'facebook', 'google', 'friend', 'course']
        df['source_clean'] = df['source'].str.lower().fillna('other')
        for i, source in enumerate(common_sources):
            df.loc[df['source_clean'].str.contains(source, na=False), 'source_encoded'] = i
        df['source_encoded'] = df['source_encoded'].fillna(len(common_sources)).astype(int)
    else:
        df['source_encoded'] = 0
    
    # 10. Generations (if available)
    if 'total_generations' not in df.columns:
        df['total_generations'] = 0
    
    # 11. Average credit per generation
    df['avg_credit_per_gen'] = np.where(
        df['total_generations'] > 0,
        df['total_credit_spent'] / df['total_generations'],
        0
    )
    
    return df

# Define feature columns
FEATURE_COLS = [
    'recency_days',
    'has_payment_failure',
    'payment_failure_count',
    'is_high_risk_card',
    'frustration_score',
    'total_credit_spent',
    'engagement_score',
    'country_code_churn_rate',
    'plan_encoded',
    'source_encoded',
    'total_generations',
    'avg_credit_per_gen'
]

# ─────────────────────────────────────────────
# INTERVENTION STRATEGY FUNCTION
# ─────────────────────────────────────────────
def get_strategy(row):
    """Generate intervention strategy based on prediction"""
    status = row['predicted_status']
    
    if status == 'invol_churn':
        if row['has_payment_failure'] == 1:
            if row.get('is_high_risk_card', 0) == 1:
                return "🔴 IMMEDIATE: Update payment method required. Offer alternative payment options (PayPal/Apple Pay). 7-day grace period."
            else:
                return "🟠 URGENT: Smart payment retry logic (3 attempts over 5 days). Send SMS reminder with payment link."
        else:
            return "🟡 PRE-EMPTIVE: High-risk pattern detected. Send verification email and update card on file."
    
    elif status == 'vol_churn':
        if row.get('frustration_score', 0) >= 3:
            if row['recency_days'] >= 14:
                return "🎁 WIN-BACK CAMPAIGN: 40% discount + 1-hour free consultation + new feature demo."
            else:
                return "💬 SUPPORT OUTREACH: Trigger in-app chat within 2 hours. Assign success manager."
        elif row['recency_days'] >= 21:
            return "📧 RE-ENGAGEMENT: 'We miss you' email sequence + showcase new features + 25% discount."
        elif row.get('engagement_score', 0) < 10:
            return "🎯 ONBOARDING BOOST: Send tutorial videos + checklist + weekly tips."
        else:
            return "⭐ VALUE DEMO: Show personalized ROI dashboard + case studies + upsell opportunity."
    
    else:  # not_churned
        if row.get('retained_confidence', 1) < 0.6:
            return "👀 WATCH LIST: Low retention confidence. Monitor for 14 days."
        else:
            return "✅ HEALTHY: Consider for loyalty program and upsell campaigns."

# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────
if 'model' not in st.session_state:
    st.session_state.model = None
if 'le' not in st.session_state:
    st.session_state.le = None
if 'df_processed' not in st.session_state:
    st.session_state.df_processed = None
if 'feature_cols' not in st.session_state:
    st.session_state.feature_cols = FEATURE_COLS

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📁 Upload & Train", "🔮 Predictions & Interventions", "📈 Executive Summary"])

# ══════════════════════════════════════════════
# TAB 1 — UPLOAD & TRAIN
# ══════════════════════════════════════════════
with tab1:
    st.header("Step 1 — Upload Your Dataset")
    
    st.markdown("""
    **Required format:** Single CSV file with all users and a `churn_status` column.
    
    Expected values for `churn_status`:
    - `not_churned` - Active users
    - `vol_churn` - Voluntary churn (user chose to cancel)
    - `invol_churn` - Involuntary churn (payment failures, technical issues)
    """)
    
    data_file = st.file_uploader("Upload your CSV file", type="csv", key="data_file")
    
    # Optional: Show sample format
    with st.expander("📋 View expected CSV format"):
        st.code("""
user_id,churn_status,subscription_plan,country_code,purchase_amount_dollars,...
user_001,not_churned,Higgsfield Pro,US,49.99,...
user_002,vol_churn,Higgsfield Basic,GB,0,...
user_003,invol_churn,Higgsfield Creator,CA,29.99,...
        """)
    
    st.markdown("---")
    st.header("Step 2 — Train the Model")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        train_button = st.button("🚀 Train Model", disabled=not data_file, use_container_width=True)
    with col2:
        if st.button("🔄 Reset Model", use_container_width=True):
            st.session_state.model = None
            st.session_state.le = None
            st.session_state.df_processed = None
            st.success("Model reset!")
    
    if train_button and data_file:
        with st.spinner("Loading and processing data..."):
            # Load data
            df_raw = pd.read_csv(data_file)
            st.success(f"✅ Loaded {len(df_raw):,} rows with {len(df_raw.columns)} columns")
            
            # Check for required column
            if 'churn_status' not in df_raw.columns:
                st.error("❌ CSV must contain a 'churn_status' column!")
                st.stop()
            
            # Show distribution
            st.markdown("### 📊 Original Churn Distribution")
            dist = df_raw['churn_status'].value_counts()
            col1, col2, col3 = st.columns(3)
            col1.metric("Not Churned", dist.get('not_churned', 0))
            col2.metric("Voluntary Churn", dist.get('vol_churn', 0))
            col3.metric("Involuntary Churn", dist.get('invol_churn', 0))
        
        with st.spinner("Engineering features..."):
            df_processed = engineer_features(df_raw)
            st.success(f"✅ Engineered {len(FEATURE_COLS)} features")
        
        with st.spinner("Training Random Forest model..."):
            # Encode target
            le = LabelEncoder()
            df_processed['target'] = le.fit_transform(df_processed['churn_status'])
            
            # Prepare features
            X = df_processed[FEATURE_COLS].fillna(0)
            y = df_processed['target']
            
            # Split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Train with class balancing
            model = RandomForestClassifier(
                n_estimators=200,
                max_depth=12,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            )
            model.fit(X_train, y_train)
            
            # Save to session
            st.session_state.model = model
            st.session_state.le = le
            st.session_state.df_processed = df_processed
            
            # Save to disk
            pickle.dump(model, open("model.pkl", "wb"))
            pickle.dump(le, open("label_encoder.pkl", "wb"))
        
        # Display metrics
        st.markdown("### 📊 Model Performance")
        
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)
        
        # Classification report
        report = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True)
        report_df = pd.DataFrame(report).transpose().round(3)
        st.dataframe(report_df, use_container_width=True)
        
        # AUC-ROC
        if len(le.classes_) == 3:
            auc_roc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
            st.metric("Multi-class AUC-ROC", f"{auc_roc:.3f}")
        
        # Confusion Matrix
        st.markdown("### 🔍 Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', ax=ax, 
                    xticklabels=le.classes_, 
                    yticklabels=le.classes_)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        st.pyplot(fig)
        
        # Feature importance
        st.markdown("### 🔑 Top Feature Importances")
        importance_df = pd.DataFrame({
            'feature': FEATURE_COLS,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        ax2.barh(importance_df['feature'][:10], importance_df['importance'][:10])
        ax2.set_xlabel('Importance')
        ax2.set_title('Top 10 Most Important Features')
        st.pyplot(fig2)
        
        st.success("🎉 Model training complete! Go to the Predictions tab to see results.")

# ══════════════════════════════════════════════
# TAB 2 — PREDICTIONS & INTERVENTIONS
# ══════════════════════════════════════════════
with tab2:
    st.header("🔮 Churn Predictions & Intervention Strategies")
    
    if st.session_state.model is None:
        st.info("👆 Please train the model first in the 'Upload & Train' tab.")
    else:
        model = st.session_state.model
        le = st.session_state.le
        df = st.session_state.df_processed
        
        # Generate predictions
        X_all = df[FEATURE_COLS].fillna(0)
        y_proba = model.predict_proba(X_all)
        y_pred = model.predict(X_all)
        
        # Create results dataframe
        results = pd.DataFrame()
        results['user_id'] = df['user_id'] if 'user_id' in df.columns else range(len(df))
        results['predicted_status'] = le.inverse_transform(y_pred)
        
        # Add probabilities for each class
        for i, class_name in enumerate(le.classes_):
            results[f'prob_{class_name}'] = y_proba[:, i]
        
        # Add key features for strategy
        results['recency_days'] = df['recency_days'].values
        results['has_payment_failure'] = df['has_payment_failure'].values
        results['payment_failure_count'] = df['payment_failure_count'].values
        results['frustration_score'] = df['frustration_score'].values
        results['is_high_risk_card'] = df['is_high_risk_card'].values
        results['engagement_score'] = df['engagement_score'].values
        results['total_credit_spent'] = df['total_credit_spent'].values
        
        # Retained confidence (probability of not_churned)
        if 'not_churned' in le.classes_:
            nc_idx = list(le.classes_).index('not_churned')
            results['retained_confidence'] = y_proba[:, nc_idx]
        else:
            results['retained_confidence'] = 0.5
        
        # Generate strategies
        results['recommended_action'] = results.apply(get_strategy, axis=1)
        
        # Calculate priority score
        max_credit = results['total_credit_spent'].max() if results['total_credit_spent'].max() > 0 else 1
        results['priority_score'] = (
            results['prob_invol_churn'].fillna(0) * 40 +
            results['prob_vol_churn'].fillna(0) * 35 +
            (results['total_credit_spent'] / max_credit) * 25
        )
        
        # Sort by priority
        results = results.sort_values('priority_score', ascending=False)
        
        # Filters
        st.markdown("### 🔍 Filter Results")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            status_filter = st.multiselect(
                "Predicted Status",
                options=['invol_churn', 'vol_churn', 'not_churned'],
                default=['invol_churn', 'vol_churn']
            )
        
        with col2:
            top_n = st.slider("Number of users to show", 10, 500, 50)
        
        with col3:
            min_risk = st.slider("Minimum risk score", 0.0, 1.0, 0.5, 0.05)
        
        # Apply filters
        filtered = results[
            (results['predicted_status'].isin(status_filter)) &
            (results[['prob_invol_churn', 'prob_vol_churn']].max(axis=1) >= min_risk)
        ].head(top_n)
        
        # Display results
        st.markdown(f"### 📋 Top {len(filtered)} Priority Users")
        
        display_cols = [
            'user_id', 'predicted_status', 'priority_score',
            'prob_invol_churn', 'prob_vol_churn', 'retained_confidence',
            'engagement_score', 'payment_failure_count', 'recommended_action'
        ]
        
        st.dataframe(filtered[display_cols].round(3), use_container_width=True)
        
        # Download buttons
        col1, col2 = st.columns(2)
        
        with col1:
            csv = filtered.to_csv(index=False)
            st.download_button(
                label="⬇️ Download Results (CSV)",
                data=csv,
                file_name="churn_predictions.csv",
                mime="text/csv"
            )
        
        with col2:
            # Summary by status
            st.markdown("### 📊 Prediction Summary")
            summary = filtered['predicted_status'].value_counts()
            st.bar_chart(summary)

# ══════════════════════════════════════════════
# TAB 3 — EXECUTIVE SUMMARY
# ══════════════════════════════════════════════
with tab3:
    st.header("📈 Executive Summary")
    
    if st.session_state.model is None:
        st.info("👆 Please train the model first in the 'Upload & Train' tab.")
    else:
        model = st.session_state.model
        le = st.session_state.le
        df = st.session_state.df_processed
        
        # Generate predictions
        X_all = df[FEATURE_COLS].fillna(0)
        y_proba = model.predict_proba(X_all)
        y_pred = model.predict(X_all)
        
        # Get indices
        classes = list(le.classes_)
        inv_idx = classes.index('invol_churn') if 'invol_churn' in classes else -1
        vol_idx = classes.index('vol_churn') if 'vol_churn' in classes else -1
        
        # KPI Cards
        st.markdown("### 🎯 Key Performance Indicators")
        
        col1, col2, col3, col4 = st.columns(4)
        
        total_users = len(df)
        high_inv_risk = (y_proba[:, inv_idx] > 0.5).sum() if inv_idx >= 0 else 0
        high_vol_risk = (y_proba[:, vol_idx] > 0.5).sum() if vol_idx >= 0 else 0
        
        at_risk_value = df[
            (y_proba[:, inv_idx] > 0.5) | (y_proba[:, vol_idx] > 0.5)
        ]['total_credit_spent'].sum() if inv_idx >= 0 or vol_idx >= 0 else 0
        
        col1.metric("👥 Total Users", f"{total_users:,}")
        col2.metric("🔴 High Involuntary Risk", f"{high_inv_risk:,}", 
                    delta=f"{(high_inv_risk/total_users*100):.1f}%" if total_users > 0 else None)
        col3.metric("🟡 High Voluntary Risk", f"{high_vol_risk:,}",
                    delta=f"{(high_vol_risk/total_users*100):.1f}%" if total_users > 0 else None)
        col4.metric("💰 Value at Risk", f"${at_risk_value:,.0f}")
        
        st.markdown("---")
        
        # Churn distribution
        st.markdown("### 📊 Churn Distribution")
        actual_dist = df['churn_status'].value_counts()
        pred_dist = pd.Series(le.inverse_transform(y_pred)).value_counts()
        
        dist_df = pd.DataFrame({
            'Actual': actual_dist,
            'Predicted': pred_dist
        }).fillna(0)
        
        st.bar_chart(dist_df)
        
        # Top risk factors
        st.markdown("### 🔑 Top Risk Factors")
        importance_df = pd.DataFrame({
            'feature': FEATURE_COLS,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(importance_df['feature'][:10], importance_df['importance'][:10])
        ax.set_xlabel('Importance Score')
        ax.set_title('Top 10 Churn Risk Factors')
        st.pyplot(fig)
        
        # Recommended actions summary
        st.markdown("### 🎯 Recommended Actions Summary")
        
        # Generate strategies for all users
        results_temp = pd.DataFrame()
        results_temp['predicted_status'] = le.inverse_transform(y_pred)
        results_temp['has_payment_failure'] = df['has_payment_failure'].values
        results_temp['is_high_risk_card'] = df['is_high_risk_card'].values
        results_temp['frustration_score'] = df['frustration_score'].values
        results_temp['recency_days'] = df['recency_days'].values
        results_temp['engagement_score'] = df['engagement_score'].values
        results_temp['retained_confidence'] = y_proba[:, list(le.classes_).index('not_churned')] if 'not_churned' in le.classes_ else 0.5
        
        results_temp['action'] = results_temp.apply(get_strategy, axis=1)
        
        action_summary = results_temp['action'].value_counts().head(10)
        st.dataframe(pd.DataFrame({'Action': action_summary.index, 'Count': action_summary.values}), 
                    use_container_width=True)
        
        # Download model files
        st.markdown("---")
        st.markdown("### 📦 Download Model Files")
        
        col1, col2 = st.columns(2)
        with col1:
            with open("model.pkl", "rb") as f:
                st.download_button(
                    label="⬇️ Download Model (model.pkl)",
                    data=f,
                    file_name="churn_model.pkl",
                    mime="application/octet-stream"
                )
        with col2:
            with open("label_encoder.pkl", "rb") as f:
                st.download_button(
                    label="⬇️ Download Label Encoder",
                    data=f,
                    file_name="label_encoder.pkl",
                    mime="application/octet-stream"
                )
