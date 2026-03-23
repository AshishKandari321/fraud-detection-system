# app.py

"""Streamlit dashboard for AI-Powered Fraud Detection in E-Governance."""
# === ADD THIS AT THE VERY TOP OF app.py ===
import os
import sys

# Ensure data directory exists for cloud deployment
if not os.path.exists("data/processed"):
    os.makedirs("data/processed", exist_ok=True)
    
# Create database if it doesn't exist
if not os.path.exists("data/processed/fraud_system.db"):
    print("🆕 First run - setting up database...")
    import database_setup
    database_setup.setup_database()
    print("✅ Database ready!")
# === END OF ADDED CODE ===

# Your existing imports continue below...
import streamlit as st
import pandas as pd
# ... rest of your imports
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import pandas as pd
import sqlite3
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import base64
from io import BytesIO

from core.pipeline import FraudDetectionPipeline
from database.schema_v2 import RelationalFraudDB


# Page config
st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5em;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 20px;
    }
    .sub-header {
        font-size: 1.2em;
        color: #666;
        text-align: center;
        margin-bottom: 30px;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
    .high-risk {
        color: #ff4b4b;
        font-weight: bold;
    }
    .medium-risk {
        color: #ffa500;
        font-weight: bold;
    }
    .low-risk {
        color: #00cc00;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)


def get_db_connection():
    """Get database connection."""
    return sqlite3.connect("data/processed/fraud_system.db")


def load_data(query, params=None):
    """Load data from database."""
    conn = get_db_connection()
    df = pd.read_sql_query(query, conn, params=params)
    conn.close()
    return df


def get_image_base64(image_path):
    """Convert image to base64."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def main():
    # Header
    st.markdown('<div class="main-header">🛡️ AI-Powered Fraud Detection</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">E-Governance Welfare Scheme Monitoring System</div>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", [
        "📊 Dashboard",
        "➕ Add Beneficiary", 
        "🔍 Single Analysis", 
        "📁 Batch Analysis",
        "📈 Analytics",
        "🕸️ Fraud Networks",
        "⚙️ Settings"
    ])
    
    if page == "📊 Dashboard":
        show_dashboard()
    elif page == "➕ Add Beneficiary":  # ADD THIS
        show_add_beneficiary()
    elif page == "🔍 Single Analysis":
        show_single_analysis()
    elif page == "📁 Batch Analysis":
        show_batch_analysis()
    elif page == "📈 Analytics":
        show_analytics()
    elif page == "🕸️ Fraud Networks":
        show_network_analysis()
    elif page == "⚙️ Settings":
        show_settings()

def show_dashboard():
    """Show main dashboard with overview statistics."""
    st.header("System Overview")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        df = load_data("SELECT COUNT(*) as count FROM beneficiaries")
        st.metric("Total Beneficiaries", f"{df['count'].iloc[0]:,}")
    
    with col2:
        df = load_data("SELECT COUNT(*) as count FROM transactions WHERE status='success'")
        st.metric("Total Transactions", f"{df['count'].iloc[0]:,}")
    
    with col3:
        df = load_data("SELECT COUNT(*) as count FROM fraud_results WHERE risk_level='High'")
        high_risk = df['count'].iloc[0]
        st.metric("High Risk Cases", f"{high_risk:,}", delta=f"{high_risk/1000:.1f}%" if high_risk > 0 else None)
    
    with col4:
        df = load_data("""
            SELECT AVG(overall_score) as avg_score 
            FROM fraud_results
        """)
        avg_score = df['avg_score'].iloc[0] if df['avg_score'].iloc[0] else 0
        st.metric("Avg Fraud Score", f"{avg_score:.1f}/100")
    
    # Charts row
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Risk Distribution")
        df = load_data("""
            SELECT risk_level, COUNT(*) as count 
            FROM fraud_results 
            GROUP BY risk_level
        """)
        if not df.empty:
            fig = px.pie(df, values='count', names='risk_level', 
                        color='risk_level',
                        color_discrete_map={
                            'High': '#ff4b4b',
                            'Medium': '#ffa500', 
                            'Low': '#00cc00'
                        })
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No analysis data yet. Run fraud detection to see results.")
    
    with col2:
        st.subheader("Scheme-wise Fraud Scores")
        # FIXED: Use transactions table for scheme_type
        df = load_data("""
            SELECT 
                t.scheme_type, 
                AVG(f.overall_score) as avg_score,
                COUNT(DISTINCT f.beneficiary_id) as beneficiary_count
            FROM fraud_results f
            JOIN transactions t ON f.beneficiary_id = t.beneficiary_id
            GROUP BY t.scheme_type
        """)
        if not df.empty:
            fig = px.bar(df, x='scheme_type', y='avg_score',
                        color='beneficiary_count',
                        title="Fraud Score by Scheme",
                        labels={'avg_score': 'Avg Fraud Score', 'scheme_type': 'Scheme'})
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Run analysis to see scheme-wise data")
    
    # Recent alerts
    st.markdown("---")
    st.subheader("🚨 Recent High-Risk Alerts")
    
    # FIXED: Remove b.scheme_type since it's in transactions table
    df = load_data("""
        SELECT 
            f.beneficiary_id,
            b.name,
            f.overall_score,
            f.risk_level,
            f.recommended_action,
            b.district,
            b.state
        FROM fraud_results f
        JOIN beneficiaries b ON f.beneficiary_id = b.beneficiary_id
        WHERE f.risk_level = 'High'
        ORDER BY f.overall_score DESC
        LIMIT 10
    """)
    
    if not df.empty:
        df['overall_score'] = df['overall_score'].round(1)
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No high-risk cases detected yet")
def show_single_analysis():
    """Analyze individual beneficiary."""
    st.header("🔍 Single Beneficiary Analysis")
    
    # Input method
    input_method = st.radio("Input Method", ["Search by ID", "Search by Aadhaar"])
    
    beneficiary_id = None  # Initialize to avoid "referenced before assignment" error
    
    if input_method == "Search by ID":
        beneficiary_id = st.text_input("Enter Beneficiary ID", "BEN00000000")
    else:
        aadhaar_input = st.text_input("Enter Aadhaar Number (12 digits)", "123456789012", max_chars=12, 
                                     help="Enter full 12-digit Aadhaar number (e.g., 123456789012)")
        
        # Lookup by Aadhaar
        if aadhaar_input:
            if len(aadhaar_input) != 12 or not aadhaar_input.isdigit():
                st.error("❌ Aadhaar must be exactly 12 digits (numbers only)")
                return
            
            from privacy.masker import DataMasker
            aadhaar_hash = DataMasker.hash_identifier(aadhaar_input)
            
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT beneficiary_id, name FROM beneficiaries WHERE aadhaar_hash = ?", (aadhaar_hash,))
            result = cursor.fetchone()
            conn.close()
            
            if result:
                beneficiary_id = result[0]
                st.success(f"✅ Found: {result[1]} (ID: {beneficiary_id})")
            else:
                st.error(f"❌ No beneficiary found with Aadhaar ending in {aadhaar_input[-4:]}")
                return
    
    if st.button("Analyze", type="primary"):
        if not beneficiary_id:
            st.error("❌ Please enter valid input first")
            return
            
        with st.spinner("Running fraud detection..."):
            try:
                pipeline = FraudDetectionPipeline()
                report = pipeline.analyze(beneficiary_id)
                display_fraud_report(report)
            except Exception as e:
                st.error(f"Error during analysis: {str(e)}")
def display_fraud_report(report):
    """Display fraud report in nice format."""
    # Score gauge
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = report.overall_score,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Fraud Score"},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 40], 'color': "lightgreen"},
                    {'range': [40, 70], 'color': "yellow"},
                    {'range': [70, 100], 'color': "salmon"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': report.overall_score
                }
            }
        ))
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    # Risk level
    risk_class = "high-risk" if report.risk_level == "High" else \
                 "medium-risk" if report.risk_level == "Medium" else "low-risk"
    
    st.markdown(f"""
        <div style="text-align: center; font-size: 1.5em; margin: 20px 0;">
            Risk Level: <span class="{risk_class}">{report.risk_level}</span>
        </div>
    """, unsafe_allow_html=True)
    
    # Engine breakdown
    st.subheader("Engine Scores")
    scores = {
        'Rule-Based (30%)': report.rule_score,
        'Velocity (25%)': report.velocity_score,
        'Graph (20%)': report.graph_score,
        'ML (15%)': report.ml_score,
        'Anomaly (10%)': report.anomaly_score
    }
    
    cols = st.columns(5)
    for i, (engine, score) in enumerate(scores.items()):
        with cols[i]:
            st.metric(engine, f"{score:.1f}")
    
    # Primary reasons
    st.subheader("⚠️ Primary Risk Indicators")
    for reason in report.primary_reasons:
        st.warning(reason)
    
    # Recommendation
    st.subheader("📋 Recommended Action")
    st.info(report.recommended_action)
    
    # Detailed indicators
    with st.expander("View Detailed Engine Outputs"):
        for indicator in report.indicators:
            st.write(f"**{indicator.engine}**: {indicator.score}/100")
            st.write(f"_{indicator.description}_")
            if indicator.details:
                st.json(indicator.details)
            st.markdown("---")


def show_batch_analysis():
    """Batch analysis page."""
    st.header("📁 Batch Fraud Analysis")
    
    uploaded_file = st.file_uploader("Upload Beneficiary CSV/Excel", 
                                    type=['csv', 'xlsx'])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else \
             pd.read_excel(uploaded_file)
        
        st.write(f"Loaded {len(df)} records")
        st.dataframe(df.head())
        
        if st.button("Run Batch Analysis", type="primary"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Initialize pipeline
            pipeline = FraudDetectionPipeline()
            
            # Analyze in batches
            results = []
            batch_size = 100
            
            for i in range(0, len(df), batch_size):
                batch = df.iloc[i:i+batch_size]
                status_text.text(f"Analyzing records {i} to {min(i+batch_size, len(df))}...")
                
                reports = pipeline.analyze_batch(batch['beneficiary_id'].tolist())
                results.extend(reports)
                
                progress = min((i + batch_size) / len(df), 1.0)
                progress_bar.progress(progress)
            
            # Convert to DataFrame
            results_df = pd.DataFrame([
                {
                    'beneficiary_id': r.beneficiary_id,
                    'overall_score': r.overall_score,
                    'risk_level': r.risk_level,
                    'rule_score': r.rule_score,
                    'velocity_score': r.velocity_score,
                    'ml_score': r.ml_score,
                    'anomaly_score': r.anomaly_score,
                    'graph_score': r.graph_score,
                    'recommended_action': r.recommended_action
                } for r in results
            ])
            
            # Merge with original data
            merged = df.merge(results_df, on='beneficiary_id', how='left')
            
            st.success(f"Analysis complete! Found {sum(merged['risk_level']=='High')} high-risk cases.")
            
            # Display results
            st.subheader("Results")
            st.dataframe(merged.sort_values('overall_score', ascending=False), 
                        use_container_width=True)
            
            # Download button
            csv = merged.to_csv(index=False)
            st.download_button(
                label="📥 Download Results CSV",
                data=csv,
                file_name=f"fraud_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )


def show_analytics():
    """Advanced analytics page."""
    st.header("📈 Advanced Analytics")
    
    # Time series (only if data exists)
    st.subheader("Fraud Trends Over Time")
    df = load_data("""
        SELECT 
            date(analyzed_at) as date,
            COUNT(*) as total_analyzed,
            AVG(overall_score) as avg_score,
            SUM(CASE WHEN risk_level='High' THEN 1 ELSE 0 END) as high_risk_count
        FROM fraud_results
        WHERE analyzed_at IS NOT NULL
        GROUP BY date(analyzed_at)
        ORDER BY date
        LIMIT 30
    """)
    
    if not df.empty and len(df) > 1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['date'], y=df['avg_score'],
                                mode='lines', name='Avg Score'))
        fig.add_trace(go.Bar(x=df['date'], y=df['high_risk_count'],
                            name='High Risk Count', yaxis='y2'))
        fig.update_layout(
            yaxis2=dict(overlaying='y', side='right'),
            title="Fraud Detection Trends"
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Run some fraud detection analyses to see trends")
    
    # FIXED: Scheme analysis using transactions table
    st.subheader("Scheme-wise Fraud Analysis")
    df = load_data("""
        SELECT 
            t.scheme_type,
            COUNT(DISTINCT f.beneficiary_id) as beneficiary_count,
            AVG(f.overall_score) as avg_fraud_score,
            SUM(CASE WHEN f.risk_level='High' THEN 1 ELSE 0 END) as high_risk_count
        FROM fraud_results f
        JOIN transactions t ON f.beneficiary_id = t.beneficiary_id
        GROUP BY t.scheme_type
    """)
    
    if not df.empty:
        fig = px.bar(df, x='scheme_type', y='avg_fraud_score',
                    color='high_risk_count',
                    title="Average Fraud Score by Scheme Type")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No fraud analysis data available yet")

def show_network_analysis():
    """Display fraud network visualizations."""
    st.header("🕸️ Fraud Network Analysis")
    
    st.info("Visualizing relationships between beneficiaries, agents, and fraud rings")
    
    # Fraud ring statistics
    df = load_data("""
        SELECT 
            a.agent_type,
            COUNT(*) as agent_count,
            AVG(a.fraud_score) as avg_fraud_score
        FROM agents a
        GROUP BY a.agent_type
    """)
    
    if not df.empty:
        st.subheader("Agent Risk Profile")
        fig = px.bar(df, x='agent_type', y='avg_fraud_score',
                    color='agent_count',
                    title="Average Fraud Score by Agent Type")
        st.plotly_chart(fig, use_container_width=True)
    
    # High-risk clusters
    st.subheader("Suspicious Clusters")
    df = load_data("""
        SELECT 
            b.address,
            COUNT(*) as beneficiary_count,
            AVG(f.overall_score) as avg_score,
            GROUP_CONCAT(b.beneficiary_id, ', ') as ids
        FROM beneficiaries b
        JOIN fraud_results f ON b.beneficiary_id = f.beneficiary_id
        GROUP BY b.address
        HAVING beneficiary_count >= 3
        ORDER BY avg_score DESC
        LIMIT 10
    """)
    
    if not df.empty:
        st.dataframe(df, use_container_width=True)


def show_settings():
    """Settings page."""
    st.header("⚙️ System Settings")
    
    st.subheader("Engine Weights")
    st.info("Adjust the weight of each detection engine in the hybrid score")
    
    col1, col2 = st.columns(2)
    
    weights = {}
    with col1:
        weights['rule'] = st.slider("Rule-Based", 0.0, 1.0, 0.30, 0.05)
        weights['velocity'] = st.slider("Velocity", 0.0, 1.0, 0.25, 0.05)
        weights['graph'] = st.slider("Graph", 0.0, 1.0, 0.20, 0.05)
    
    with col2:
        weights['ml'] = st.slider("ML", 0.0, 1.0, 0.15, 0.05)
        weights['anomaly'] = st.slider("Anomaly", 0.0, 1.0, 0.10, 0.05)
    
    total = sum(weights.values())
    st.write(f"**Total Weight: {total:.2f}** {'✓ Valid' if abs(total - 1.0) < 0.01 else '⚠️ Must sum to 1.0'}")
    
    # Database stats
    st.markdown("---")
    st.subheader("Database Statistics")
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    tables = ['beneficiaries', 'agents', 'transactions', 'fraud_results']
    for table in tables:
        cursor.execute(f"SELECT COUNT(*) FROM {table}")
        count = cursor.fetchone()[0]
        st.write(f"{table}: {count:,} records")
    
    conn.close()
def show_add_beneficiary():
    """Add new beneficiary with scheme selection."""
    st.header("➕ Register New Beneficiary")
    st.info("Register beneficiary for specific welfare scheme with fraud detection")
    
    with st.form("add_beneficiary_form"):
        st.subheader("Personal Information")
        col1, col2 = st.columns(2)
        
        with col1:
            beneficiary_id = st.text_input("Beneficiary ID*", f"BEN{99996:08d}")
            name = st.text_input("Full Name*", "Rajesh Kumar")
            aadhaar = st.text_input("Aadhaar (12 digits)*", "123456789012", max_chars=12)
            phone = st.text_input("Phone (10 digits)*", "9876543210", max_chars=10)
            address = st.text_area("Address*", "123 Main Street, Delhi")
        
        with col2:
            # SCHEME SELECTION - Critical for fraud detection
            scheme_type = st.selectbox(
                "Welfare Scheme*",
                ["PDS", "PAHAL", "PM_KISAN", "PENSION", "SCHOLARSHIP"],
                help="Select the scheme beneficiary is registering for"
            )
            
            annual_income = st.number_input("Annual Income (₹)*", 0, 5000000, 250000, 
                                           help="Income eligibility varies by scheme")
            family_size = st.number_input("Family Size*", 1, 15, 4)
            district = st.text_input("District*", "New Delhi")
            state = st.selectbox("State*", [
                "Delhi", "Maharashtra", "Uttar Pradesh", "Bihar", "West Bengal",
                "Tamil Nadu", "Karnataka", "Gujarat", "Rajasthan", "Punjab"
            ])
            pincode = st.text_input("Pincode*", "110001", max_chars=6)
        
        st.subheader("Bank Details")
        col3, col4 = st.columns(2)
        with col3:
            bank_account = st.text_input("Bank Account Number*", "SBIN0123456789")
        with col4:
            ifsc_code = st.text_input("IFSC Code*", "SBIN0001234")
        
        # Eligibility warning based on scheme
        st.subheader("Eligibility Check")
        if scheme_type == "PDS" and annual_income > 300000:
            st.warning("⚠️ Income exceeds PDS BPL threshold (₹3 Lakhs). May be flagged as ineligible.")
        elif scheme_type == "PENSION" and annual_income > 200000:
            st.warning("⚠️ Income exceeds Pension eligibility (₹2 Lakhs). Likely fraud risk.")
        elif scheme_type == "SCHOLARSHIP" and annual_income > 800000:
            st.warning("⚠️ Income exceeds Scholarship limit (₹8 Lakhs). High fraud risk.")
        
        submit = st.form_submit_button("🚀 Register & Analyze", use_container_width=True)
    
    if submit:
        from privacy.masker import DataMasker
        from datetime import date
        
        # Validation
        if len(aadhaar) != 12 or not aadhaar.isdigit():
            st.error("❌ Aadhaar must be 12 digits")
            return
        if len(phone) != 10 or not phone.isdigit():
            st.error("❌ Phone must be 10 digits")
            return
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        try:
            # Check duplicate ID
            cursor.execute("SELECT 1 FROM beneficiaries WHERE beneficiary_id = ?", (beneficiary_id,))
            if cursor.fetchone():
                st.error(f"❌ ID {beneficiary_id} already exists!")
                return
            
            # Check duplicate Aadhaar
            aadhaar_hash = DataMasker.hash_identifier(aadhaar)
            cursor.execute("SELECT beneficiary_id FROM beneficiaries WHERE aadhaar_hash = ?", (aadhaar_hash,))
            existing = cursor.fetchone()
            if existing:
                st.error(f"🚨 FRAUD ALERT: Duplicate Aadhaar! Existing: {existing[0]}")
                st.info("Registration blocked due to duplicate identity.")
                return
            
            # Insert with scheme_type
            cursor.execute("""
                INSERT INTO beneficiaries 
                (beneficiary_id, aadhaar_hash, aadhaar_masked, name, address, 
                 phone_hash, phone_masked, bank_hash, bank_masked, ifsc_code,
                 annual_income, scheme_type, district, state, pincode, 
                 family_size, registration_date, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                beneficiary_id, aadhaar_hash, DataMasker.mask_aadhaar(aadhaar),
                name, address, DataMasker.hash_identifier(phone), DataMasker.mask_phone(phone),
                DataMasker.hash_identifier(bank_account), DataMasker.mask_bank_account(bank_account),
                ifsc_code, annual_income, scheme_type, district, state, pincode,
                family_size, date.today().isoformat(), 'active'
            ))
            
            conn.commit()
            st.success(f"✅ Registered {name} for {scheme_type} scheme!")
            
            # Immediate fraud check
            with st.spinner("Running fraud detection..."):
                pipeline = FraudDetectionPipeline()
                report = pipeline.analyze(beneficiary_id)
                
                st.markdown("---")
                st.subheader("🔍 Fraud Analysis Result")
                
                # Color-coded risk
                if report.risk_level == "High":
                    st.error(f"🚨 HIGH RISK (Score: {report.overall_score:.1f}/100)")
                elif report.risk_level == "Medium":
                    st.warning(f"⚠️ MEDIUM RISK (Score: {report.overall_score:.1f}/100)")
                else:
                    st.success(f"✓ LOW RISK (Score: {report.overall_score:.1f}/100)")
                
                # Show why
                if report.primary_reasons:
                    st.write("**Key Findings:**")
                    for reason in report.primary_reasons:
                        st.write(f"- {reason}")
                
                # Scheme-specific alert
                if "income" in str(report.primary_reasons).lower():
                    st.error(f"**Income Mismatch:** This beneficiary's income (₹{annual_income:,}) may not be eligible for {scheme_type} scheme!")
                
        except Exception as e:
            st.error(f"❌ Database error: {str(e)}")
            conn.rollback()
        finally:
            conn.close()


if __name__ == "__main__":
    main()