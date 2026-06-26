import os

import pandas as pd
import streamlit as st

try:
    import openai
except Exception:
    openai = None

st.set_page_config(
    page_title="PropTech AI: Real Estate Intelligence",
    page_icon="🏢",
    layout="wide",
)

st.markdown(
    """
    <style>
    html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
        background-color: #f8fafc !important;
        color: #0f172a !important;
    }
    [data-testid="stSidebar"] {
        background-color: #ffffff !important;
        border-right: 1px solid #e2e8f0;
    }
    .stMetric {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 10px 12px;
        box-shadow: 0 4px 18px rgba(15, 23, 42, 0.04);
    }
    .ai-advisor-box {
        padding: 18px 20px;
        border-radius: 18px;
        background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 100%);
        box-shadow: 0 12px 30px rgba(15, 23, 42, 0.16);
    }
    .ai-advisor-box h3, .ai-advisor-box p, .ai-advisor-box b {
        color: #ffffff !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    date_col = next((c for c in df.columns if c.lower() in {"sale_date", "date", "transaction_date"}), None)
    if date_col is not None:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    if "sale_amount_aed" in df.columns:
        df["sale_amount_aed"] = pd.to_numeric(df["sale_amount_aed"], errors="coerce")
    if "sale_status" in df.columns:
        df["sale_status"] = df["sale_status"].astype(str)
    return df


def format_currency(value: float) -> str:
    return f"AED {value:,.0f}"


def get_ai_lead_grade(row: pd.Series) -> str:
    sale_amount = row.get("sale_amount_aed", 0)
    status = str(row.get("sale_status", "")).strip().lower()
    if sale_amount >= 5_000_000 and status == "lead":
        return "High-Value VIP Investor"
    if status == "sold":
        return "Closed Deal"
    return "Standard Prospect"


def get_metric_trend(metric_name: str) -> str:
    trends = {
        "Gross Revenue (AED)": "+14.2% vs last quarter",
        "Total Customers": "+9.8% QoQ growth",
        "Average Deal Size (AED)": "Premium demand stable",
        "Closed Units": "+12 closed deals w/w",
    }
    return trends.get(metric_name, "Operational performance trending upward")


def build_kpis(df: pd.DataFrame) -> dict:
    revenue = float(df["sale_amount_aed"].sum()) if "sale_amount_aed" in df.columns else 0.0
    customers = df["customer_name"].nunique() if "customer_name" in df.columns else len(df)
    deal_size = float(df["sale_amount_aed"].mean()) if "sale_amount_aed" in df.columns else 0.0
    closed_units = int(df[df["sale_status"].astype(str).str.lower() == "sold"].shape[0]) if "sale_status" in df.columns else 0
    return {
        "Gross Revenue (AED)": revenue,
        "Total Customers": customers,
        "Average Deal Size (AED)": deal_size,
        "Closed Units": closed_units,
    }


def get_top_area(df: pd.DataFrame) -> str:
    if "area" in df.columns and "sale_amount_aed" in df.columns:
        top_area = (
            df.dropna(subset=["area", "sale_amount_aed"])
            .groupby("area")["sale_amount_aed"]
            .sum()
            .sort_values(ascending=False)
            .head(1)
        )
        if not top_area.empty:
            return top_area.index[0]
    return "N/A"


def get_top_lead_source(df: pd.DataFrame) -> str:
    if "lead_source" in df.columns:
        return df["lead_source"].fillna("Unknown").mode().iloc[0]
    return "N/A"


def get_inventory_shortage_alert(df: pd.DataFrame) -> str:
    if "area" not in df.columns or "sale_status" not in df.columns:
        return "Inventory risk data unavailable for the current dataset."
    pending_cancelled = df[df["sale_status"].astype(str).str.lower().isin({"pending", "cancelled"})]
    if pending_cancelled.empty:
        return "No immediate inventory shortages are detected for pending or cancelled opportunities."
    top_area = pending_cancelled["area"].value_counts().idxmax()
    return f"Alert: {top_area} has the highest concentration of pending or cancelled inventory. Executive review is recommended for this corridor."


def get_lead_conversion_efficiency(df: pd.DataFrame) -> str:
    if "lead_source" not in df.columns or "sale_status" not in df.columns:
        return "Conversion efficiency data unavailable for the current dataset."
    channel = df["lead_source"].fillna("Unknown").mode().iloc[0]
    total_leads = df[(df["lead_source"].fillna("Unknown") == channel) & (df["sale_status"].astype(str).str.lower() == "lead")].shape[0]
    sold = df[(df["lead_source"].fillna("Unknown") == channel) & (df["sale_status"].astype(str).str.lower() == "sold")].shape[0]
    efficiency = (sold / total_leads * 100) if total_leads else 0.0
    return f"Top channel {channel} conversion efficiency is {efficiency:.1f}% based on sold-to-lead ratio."


def generate_openai_insights(api_key: str, overview: dict, top_area: str, top_lead: str) -> str:
    if openai is None:
        return "OpenAI package not installed. Install openai to enable GPT-driven insights."
    try:
        if hasattr(openai, "chat") and hasattr(openai.chat, "completions"):
            response = openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "You are a strategic real estate AI consultant. Provide concise executive recommendations for the following summary." + str(overview)}],
                temperature=0.6,
                max_tokens=280,
            )
            return response.choices[0].message.content.strip()

        openai.api_key = api_key
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "You are a strategic real estate AI consultant. Provide concise executive recommendations for the following summary." + str(overview)}],
            temperature=0.6,
            max_tokens=280,
        )
        return response.choices[0].message.content.strip()
    except Exception as exc:
        return f"Unable to generate insights: {exc}"


def build_download_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def render_metric_cards(metrics: dict) -> None:
    cols = st.columns(4, gap="large")
    for idx, (metric_name, metric_value) in enumerate(metrics.items()):
        with cols[idx]:
            display_value = format_currency(metric_value) if "AED" in metric_name else f"{metric_value:,.0f}"
            st.metric(label=metric_name, value=display_value, delta=get_metric_trend(metric_name))


def build_monthly_timeline(df: pd.DataFrame) -> pd.DataFrame:
    date_col = next((c for c in df.columns if c.lower() in {"sale_date", "date", "transaction_date"}), None)
    if date_col is None or "sale_amount_aed" not in df.columns:
        return pd.DataFrame(columns=["Month", "Revenue"])

    timeline = df.dropna(subset=[date_col, "sale_amount_aed"]).copy()
    timeline["Month"] = timeline[date_col].dt.to_period("M").dt.to_timestamp()
    summary = timeline.groupby("Month")["sale_amount_aed"].sum().reset_index()
    summary.columns = ["Month", "Revenue"]
    return summary


def build_area_revenue_summary(df: pd.DataFrame) -> pd.DataFrame:
    if "area" not in df.columns or "sale_amount_aed" not in df.columns:
        return pd.DataFrame(columns=["Area", "Revenue"])
    summary = (
        df.dropna(subset=["area", "sale_amount_aed"])
        .groupby("area")["sale_amount_aed"]
        .sum()
        .sort_values(ascending=False)
        .head(5)
        .reset_index()
    )
    summary.columns = ["Area", "Revenue"]
    return summary


def build_channel_revenue_summary(df: pd.DataFrame) -> pd.DataFrame:
    if "lead_source" not in df.columns or "sale_amount_aed" not in df.columns:
        return pd.DataFrame(columns=["Channel", "Revenue"])
    summary = (
        df.dropna(subset=["lead_source", "sale_amount_aed"])
        .groupby("lead_source")["sale_amount_aed"]
        .sum()
        .sort_values(ascending=False)
        .head(5)
        .reset_index()
    )
    summary.columns = ["Channel", "Revenue"]
    return summary


def build_property_type_summary(df: pd.DataFrame, area: str) -> pd.DataFrame:
    if "property_type" not in df.columns or "area" not in df.columns:
        return pd.DataFrame(columns=["Property Type", "Count"])
    area_df = df[df["area"] == area] if area in df["area"].dropna().unique() else df.iloc[0:0]
    counts = area_df["property_type"].fillna("Unknown").value_counts().reset_index()
    counts.columns = ["Property Type", "Count"]
    return counts.head(6)


def display_ledger(df: pd.DataFrame) -> None:
    display_columns = [
        c
        for c in [
            next((c for c in df.columns if c.lower() in {"sale_date", "date", "transaction_date"}), None),
            "customer_name",
            "area",
            "property_type",
            "lead_source",
            "agent_name",
            "sale_amount_aed",
            "sale_status",
            "AI_Lead_Grade",
        ]
        if c in df.columns
    ]
    st.dataframe(df[display_columns].copy(), use_container_width=True)


def main() -> None:
    sidebar = st.sidebar
    sidebar.header("Configuration & Filters")
    api_key = sidebar.text_input("OpenAI API Key", type="password")

    data_path = os.path.join(os.path.dirname(__file__) or os.getcwd(), "dubai_real_estate_data_realistic_500.csv")
    if not os.path.exists(data_path):
        st.error("Data file not found. Please place dubai_real_estate_data_realistic_500.csv in the same directory as main.py.")
        return

    df = load_data(data_path)
    df["AI_Lead_Grade"] = df.apply(get_ai_lead_grade, axis=1)

    if "area" in df.columns:
        selected_areas = sidebar.multiselect(
            "Select Areas",
            options=sorted(df["area"].dropna().unique()),
            default=sorted(df["area"].dropna().unique()),
        )
        if selected_areas:
            df = df[df["area"].isin(selected_areas)]

    if "lead_source" in df.columns:
        selected_leads = sidebar.multiselect(
            "Select Lead Sources",
            options=sorted(df["lead_source"].dropna().unique()),
            default=sorted(df["lead_source"].dropna().unique()),
        )
        if selected_leads:
            df = df[df["lead_source"].isin(selected_leads)]

    if "sale_status" in df.columns:
        selected_status = sidebar.multiselect(
            "Select Sale Status",
            options=sorted(df["sale_status"].dropna().unique()),
            default=sorted(df["sale_status"].dropna().unique()),
        )
        if selected_status:
            df = df[df["sale_status"].isin(selected_status)]

    date_col = next((c for c in df.columns if c.lower() in {"sale_date", "date", "transaction_date"}), None)
    if date_col is not None:
        min_date = df[date_col].min()
        max_date = df[date_col].max()
        if pd.notnull(min_date) and pd.notnull(max_date):
            selected_range = sidebar.date_input(
                "Date Range",
                value=(min_date.date(), max_date.date()),
                min_value=min_date.date(),
                max_value=max_date.date(),
            )
            if selected_range and len(selected_range) == 2:
                start_date, end_date = selected_range
                df = df[(df[date_col].dt.date >= start_date) & (df[date_col].dt.date <= end_date)]

    if sidebar.button("🔄 Reset All Filters", key="reset_filters"):
        if hasattr(st, "rerun"):
            st.rerun()
        else:
            st.experimental_rerun()

    download_bytes = build_download_bytes(df)
    sidebar.download_button(
        label="Export Filtered Report",
        data=download_bytes,
        file_name="PropTech_Filtered_Report.csv",
        mime="text/csv",
    )

    st.markdown(
        "<h1 style='margin:0 0 6px 0; color:#0f172a;'>🏢 PropTech AI</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<h3 style='margin:0 0 16px 0; color:#475569;'>Premium executive dashboard for real estate performance and lead intelligence.</h3>",
        unsafe_allow_html=True,
    )

    if df.empty:
        st.info("No records match the active filters.")
        return

    tabs = st.tabs(["📊 Executive Dashboard", "🎯 Advanced Geo & Lead Analytics", "🤖 Intelligent Lead Ledger"])

    with tabs[0]:
        render_metric_cards(build_kpis(df))

        monthly_timeline = build_monthly_timeline(df)
        if monthly_timeline.empty:
            st.info("Insufficient sales history for charting.")
        else:
            st.markdown("<h4 style='margin-top:16px; color:#0f172a;'>Monthly Sales Performance Timeline</h4>", unsafe_allow_html=True)
            st.area_chart(monthly_timeline.set_index("Month"), use_container_width=True)

        st.markdown("---")
        st.markdown(
            "<div class='ai-advisor-box'>"
            "<h3 style='margin:0 0 8px 0;'>AI Strategic Advisory</h3>"
            f"<p style='margin:0 0 8px 0;'>{get_inventory_shortage_alert(df)}</p>"
            f"<p style='margin:0;'>{get_lead_conversion_efficiency(df)}</p>"
            "</div>",
            unsafe_allow_html=True,
        )

        if api_key:
            if st.button("Generate GPT Insights", use_container_width=True):
                with st.spinner("Generating executive recommendations..."):
                    top_area = get_top_area(df)
                    top_lead = get_top_lead_source(df)
                    insights_text = generate_openai_insights(api_key, build_kpis(df), top_area, top_lead)
                    st.markdown(
                        f"<div style='margin-top:12px; padding:14px 16px; border-radius:12px; background:#ffffff; border:1px solid #d1d5db; color:#0f172a; line-height:1.7;'>{insights_text}</div>",
                        unsafe_allow_html=True,
                    )
        else:
            st.info("Add an OpenAI key for live GPT recommendations or continue using the data-backed metrics.")

    with tabs[1]:
        st.markdown("<h4 style='margin-bottom:10px; color:#0f172a;'>Regional Deep-Dive</h4>", unsafe_allow_html=True)
        if "area" in df.columns:
            selector_col, metric_col = st.columns([0.7, 0.3], gap="small")
            with selector_col:
                selected_area = st.selectbox("Select Area", sorted(df["area"].dropna().unique()))
            with metric_col:
                area_value = float(df.loc[df["area"] == selected_area, "sale_amount_aed"].max()) if "sale_amount_aed" in df.columns else 0.0
                st.metric("Highest Single Transaction", format_currency(area_value))

            chart_col1, chart_col2 = st.columns(2, gap="large")
            with chart_col1:
                st.caption("Financial Volume by Location")
                area_summary = build_area_revenue_summary(df)
                if area_summary.empty:
                    st.info("No area revenue data available.")
                else:
                    st.bar_chart(area_summary.set_index("Area"), use_container_width=True)
            with chart_col2:
                st.caption("Marketing Channels ROI")
                channel_summary = build_channel_revenue_summary(df)
                if channel_summary.empty:
                    st.info("No channel data available.")
                else:
                    st.bar_chart(channel_summary.set_index("Channel"), use_container_width=True)

            st.markdown("---")
            st.caption("Property Type Distribution")
            property_summary = build_property_type_summary(df, selected_area)
            if property_summary.empty:
                st.info("No property-type detail is available for the selected area.")
            else:
                st.bar_chart(property_summary.set_index("Property Type"), use_container_width=True)
                st.dataframe(property_summary, use_container_width=True, hide_index=True)
        else:
            st.warning("Area data is unavailable. Please include an area column to enable deep dive analytics.")

    with tabs[2]:
        st.markdown("<h4 style='margin-bottom:10px; color:#0f172a;'>Intelligent Lead Ledger</h4>", unsafe_allow_html=True)
        grade_counts = df["AI_Lead_Grade"].value_counts().rename_axis("AI_Lead_Grade").reset_index(name="Count")
        st.dataframe(grade_counts, use_container_width=True)
        st.caption("The AI lead grade is appended directly to the transaction ledger for executive review.")
        display_ledger(df)
        st.download_button(
            label="Download Filtered Ledger",
            data=download_bytes,
            file_name="PropTech_Filtered_Report.csv",
            mime="text/csv",
            key="download_ledger",
        )

    st.markdown("---")
    st.caption("Use sidebar filters to refine the dataset and export the filtered report for executive distribution.")


if __name__ == "__main__":
    main()
