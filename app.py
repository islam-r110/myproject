import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from openai import OpenAI

# ==========================
# Page Configuration
# ==========================
st.set_page_config(
    page_title="PropTech AI Intelligence Platform",
    page_icon="👑",
    layout="wide"
)

# ==========================
# VIP Custom Styling (Clean & Stable Customization)
# ==========================
st.markdown("""
<style>
    .main-title {
        font-size: 32px;
        font-weight: bold;
        color: #0f172a;
        font-family: 'Segoe UI', Roboto, sans-serif;
        margin-bottom: 5px;
    }
    .ai-container {
        background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 100%);
        color: white;
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0px 4px 15px rgba(30, 58, 138, 0.15);
        margin-bottom: 25px;
    }
    .guide-box {
        background-color: #f1f5f9;
        padding: 25px;
        border-radius: 12px;
        border-left: 5px solid #3b82f6;
        margin-top: 30px;
    }
</style>
""", unsafe_allow_html=True)

# ==========================
# Initialize OpenAI Client
# ==========================
AI_KEY = st.sidebar.text_input("OpenAI API Key (Optional for Live Insights)", type="password")
client = OpenAI(api_key=AI_KEY) if AI_KEY else None

# ==========================
# Load Data
# ==========================
@st.cache_data
def load_data():
    df = pd.read_csv("dubai_real_estate_data_realistic_500.csv")
    df["purchase_date"] = pd.to_datetime(df["purchase_date"])
    return df

try:
    df = load_data()
except FileNotFoundError:
    st.error("⚠️ Data file 'dubai_real_estate_data_realistic_500.csv' not found. Please check the path.")
    st.stop()

# Basic Data Cleaning
df = df.drop_duplicates().dropna()
for col in df.select_dtypes(include="object").columns:
    df[col] = df[col].str.strip()

# ==========================
# Sidebar Filters
# ==========================
st.sidebar.markdown("### 🎛️ Control Panel & Filters")
st.sidebar.markdown("---")

areas = st.sidebar.multiselect("📍 Filter by Area", options=sorted(df["area"].unique()), default=sorted(df["area"].unique()))
sources = st.sidebar.multiselect("📢 Filter by Lead Source", options=sorted(df["lead_source"].unique()), default=sorted(df["lead_source"].unique()))
statuses = st.sidebar.multiselect("💼 Filter by Sale Status", options=sorted(df["sale_status"].unique()), default=sorted(df["sale_status"].unique()))

if not areas or not sources or not statuses:
    st.warning("⚠️ Please select at least one item from each filter to render data.")
    st.stop()

filtered_df = df[
    (df["area"].isin(areas)) & 
    (df["lead_source"].isin(sources)) & 
    (df["sale_status"].isin(statuses))
]

total_sales = filtered_df["sale_amount_aed"].sum()

# ==========================
# Header Component
# ==========================
st.markdown('<p class="main-title">🏢 PropTech AI: Real Estate Intelligence Platform</p>', unsafe_allow_html=True)
st.markdown("<p style='color: #64748b; margin-bottom: 25px;'>Advanced Background Automation & Generative AI for Enterprise Brokerages</p>", unsafe_allow_html=True)

# ==========================
# AI Live Executive Summary
# ==========================
st.markdown('<div class="ai-container">', unsafe_allow_html=True)
st.markdown("<h3 style='color: white !important; margin-top:0;'>🤖 AI Executive Consultant Insights</h3>", unsafe_allow_html=True)

if client and not filtered_df.empty:
    if st.button("🔄 Generate Live Strategic Analysis via GPT"):
        with st.spinner("Analyzing active parameters and formulating advisory..."):
            summary_metrics = filtered_df.groupby("area")["sale_amount_aed"].sum().to_string()
            source_metrics = filtered_df.groupby("lead_source")["sale_amount_aed"].sum().to_string()
            
            prompt = f"""
            You are an expert real estate business consultant in Dubai. Analyze these metrics and provide a professional executive report in English in exactly 3 bullet points (Current State, Best Marketing Channel, and actionable Advice for the CEO to maximize profit):
            Sales by Area:
            {summary_metrics}
            Sales by Lead Source:
            {source_metrics}
            Total Sales: {total_sales} AED.
            """
            try:
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}]
                )
                ai_analysis = response.choices[0].message.content
                st.markdown(f"<div style='background: rgba(255,255,255,0.1); padding: 15px; border-radius: 10px; margin-top: 15px; color: white !important;'>{ai_analysis}</div>", unsafe_allow_html=True)
            except Exception as e:
                st.error(f"OpenAI API Connection Error: {e}")
else:
    best_source = filtered_df.groupby("lead_source")["sale_amount_aed"].sum().idxmax() if not filtered_df.empty else "N/A"
    top_area = filtered_df.groupby("area")["sale_amount_aed"].sum().idxmax() if not filtered_df.empty else "N/A"
        
    st.markdown(f"""
    <div style="padding: 5px 0px;">
        <p style="font-size: 16px; color: #ffffff !important; margin-bottom: 12px;">
            <span style="color: #f59e0b; font-weight: bold;">• Capital Reallocation:</span> Current data signals heavy market velocity in <span style="color: #f59e0b; font-weight: bold;">{top_area}</span>. We strongly advise focusing senior consultants on available inventory in this sector.
        </p>
        <p style="font-size: 16px; color: #ffffff !important;">
            <span style="color: #f59e0b; font-weight: bold;">• Marketing Spend Efficiency:</span> <span style="color: #f59e0b; font-weight: bold;">{best_source}</span> generates the highest transactional ROI. Review and downscale budget on lower-performing channels to instantly cut customer acquisition costs (CAC).
        </p>
    </div>
    """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# ==========================
# Metric Cards Row (Using Native Streamlit Metrics to guarantee stability)
# ==========================
total_sales = filtered_df["sale_amount_aed"].sum()
avg_sale = filtered_df["sale_amount_aed"].mean() if not filtered_df.empty else 0
total_customers = filtered_df["customer_id"].nunique()
sold_properties = (filtered_df["sale_status"] == "Sold").sum()

kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
with kpi_col1:
    st.metric(label="Gross Revenue (AED)", value=f"{total_sales:,.0f}")
with kpi_col2:
    st.metric(label="Total Customers", value=f"{total_customers:,}")
with kpi_col3:
    st.metric(label="Average Deal Size (AED)", value=f"{avg_sale:,.0f}")
with kpi_col4:
    st.metric(label="Closed Units", value=f"{sold_properties:,}")

st.write("")
st.divider()

# ==========================
# Advanced Charts Section
# ==========================
sns.set_theme(style="white")
corporate_colors = ["#0f172a", "#1e3a8a", "#3b82f6", "#475569", "#94a3b8"]

if not filtered_df.empty:
    row1_col1, row1_col2 = st.columns(2)
    
    with row1_col1:
        st.markdown("#### 📍 Financial Volume by Geographic Location (Areas)")
        area_sales = filtered_df.groupby("area")["sale_amount_aed"].sum().sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.barplot(x=area_sales.values, y=area_sales.index, palette=corporate_colors, ax=ax)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        st.pyplot(fig)
        plt.close()

    with row1_col2:
        st.markdown("#### 📢 Marketing Channels ROI (Ad Spend Performance)")
        source_sales = filtered_df.groupby("lead_source")["sale_amount_aed"].sum().sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.barplot(x=source_sales.index, y=source_sales.values, palette=corporate_colors, ax=ax)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        st.pyplot(fig)
        plt.close()

    st.write("")
    row2_col1, row2_col2 = st.columns(2)

    with row2_col1:
        st.markdown("#### 📈 Monthly Sales Performance Timeline")
        monthly_sales = filtered_df.groupby(filtered_df["purchase_date"].dt.to_period("M"))["sale_amount_aed"].sum()
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(monthly_sales.index.astype(str), monthly_sales.values, marker="o", color="#1e3a8a", linewidth=2.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.xticks(rotation=45)
        st.pyplot(fig)
        plt.close()

    with row2_col2:
        st.markdown("#### 🏆 Top 5 Performing Real Estate Consultants")
        agent_sales = filtered_df.groupby("agent_name")["sale_amount_aed"].sum().sort_values(ascending=False).head(5)
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.barplot(x=agent_sales.values, y=agent_sales.index, palette=corporate_colors, ax=ax)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        st.pyplot(fig)
        plt.close()

# ==========================
# AI Lead Scoring Automation & Logs
# ==========================
def score_lead(row):
    if row['sale_amount_aed'] >= 5000000 and row['sale_status'] == 'Lead':
        return "🔥 High-Value VIP Investor"
    elif row['sale_status'] == 'Sold':
        return "✅ Closed Deal"
    else:
        return "⚡ Standard Prospect"

filtered_df['AI_Lead_Grade'] = filtered_df.apply(score_lead, axis=1)

st.divider()
st.markdown("#### 📋 Consolidated Transaction Log & AI Lead Scoring Engine")

display_cols = ['customer_id', 'nationality', 'area', 'property_type', 'sale_amount_aed', 'lead_source', 'sale_status', 'AI_Lead_Grade']
st.dataframe(filtered_df[display_cols], use_container_width=True)

# ==========================
# User Guide & Documentation
# ==========================
st.divider()
st.markdown('<div class="guide-box">', unsafe_allow_html=True)
st.markdown("### 📘 Platform Deployment & User Guide")
st.markdown("""
Welcome to the **PropTech AI Intelligence Platform**. This dashboard is engineered to convert raw transactional data into bottom-line profitability. Here is how to navigate and extract value from the system:

1. **How to use Filters:** Use the **Control Panel** on the left sidebar to isolate specific zones (e.g., *Palm Jumeirah*), advertising nodes (e.g., *Instagram*), or conversion milestones (*Sold, Pending*). The entire system, KPIs, and graphs recalculate instantly.
2. **AI Executive Consultant:** The dark top panel simulates real-time strategic consultancy. If you input your secure OpenAI Key in the sidebar, clicking the refresh button will command an LLM to evaluate your actual dataset values and write live strategic directives.
3. **AI Lead Scoring Engine:** Look at the far-right column of the data ledger. The system processes customer purchasing capabilities in the background, automatically appending a `🔥 High-Value VIP Investor` badge next to users with massive capital potential who haven't finalized a purchase yet, signaling immediate sales prioritization.
4. **Data Synchronization:** To upload a new data ledger, replace the background source file (`dubai_real_estate_data_realistic_500.csv`) with the updated sheet using identical headers. The pipeline automates cleaning, typecasting, and processing autonomously.
""")
st.markdown('</div>', unsafe_allow_html=True)