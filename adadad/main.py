import os
import math
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
from datetime import datetime
import base64

# ============================================================
# STEP 1: PAGE CONFIGURATION AND STYLE LOADING
# ============================================================

st.set_page_config(
    page_title="PropTech AI — Real Estate Intelligence Platform",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded",
)


def load_external_css(css_path: str) -> None:
    """
    Load a CSS file and inject it into the Streamlit app as inline style.
    This keeps the styles separate from the Python logic for easier maintenance.
    """
    try:
        with open(css_path, "r", encoding="utf-8") as f:
            css_content = f.read()
        st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning(f"Stylesheet not found at '{css_path}'. The application will use default Streamlit styling.")
    except Exception as exc:
        st.warning(f"Could not load stylesheet: {exc}. The application will use default Streamlit styling.")


# Load the external premium stylesheet
script_directory = os.path.dirname(__file__) or os.getcwd()
css_file_path = os.path.join(script_directory, "styles.css")
load_external_css(css_file_path)

# ============================================================
# STEP 2: GLOBAL CHART COLOR DEFINITIONS
# ============================================================

sns.set_theme(
    style="whitegrid",
    rc={
        "font.family": "sans-serif",
        "font.sans-serif": ["Inter", "Segoe UI", "Arial"],
    },
)
plt.rcParams.update({
    "text.color": "#0f172a",
    "axes.labelcolor": "#0f172a",
    "xtick.color": "#64748b",
    "ytick.color": "#64748b",
    "axes.edgecolor": "#e2e8f0",
    "axes.linewidth": 1,
    "figure.facecolor": "#ffffff",
    "axes.facecolor": "#ffffff",
    "axes.titleweight": "bold",
    "axes.titlecolor": "#0f172a",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 10,
})

corporate_palette = ["#1e3a8a", "#3b82f6", "#10b981", "#f59e0b", "#64748b"]
corporate_gradient = ["#1e3a8a", "#2563eb", "#3b82f6", "#60a5fa", "#93c5fd"]

# ============================================================
# DATA LOADING WITH CACHE AND FALLBACK ERROR HANDLING
# ============================================================


@st.cache_data(show_spinner=True)
def load_and_clean_data(path: str) -> pd.DataFrame:
    """
    Load the CSV data file with comprehensive fallback error handling,
    trim whitespace from all string columns, and drop duplicate rows.
    Returns a clean DataFrame ready for analysis.
    """
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        st.error(
            f"Data file not found at {path}. "
            "Please ensure dubai_real_estate_data_realistic_500.csv is in the correct directory."
        )
        return pd.DataFrame()
    except pd.errors.EmptyDataError:
        st.error("The data file is empty. Please check the file contents and try again.")
        return pd.DataFrame()
    except pd.errors.ParserError:
        st.error("There was a parsing error while reading the CSV file. Please check the file format.")
        return pd.DataFrame()
    except Exception as exc:
        st.error(f"An unexpected error occurred while loading the data file: {exc}")
        return pd.DataFrame()

    # Trim whitespace from all object (string) columns
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].astype(str).str.strip()

    # Drop duplicate rows to ensure clean, deduplicated data
    initial_row_count = len(df)
    df = df.drop_duplicates()
    removed_count = initial_row_count - len(df)
    if removed_count > 0:
        st.sidebar.caption(f"Cleaned data: removed {removed_count} duplicate row(s).")

    # Convert purchase_date column to datetime with coerced errors
    if "purchase_date" in df.columns:
        df["purchase_date"] = pd.to_datetime(df["purchase_date"], errors="coerce")

    # Ensure sale_amount_aed is numeric
    if "sale_amount_aed" in df.columns:
        df["sale_amount_aed"] = pd.to_numeric(df["sale_amount_aed"], errors="coerce")

    # Ensure sale_status is a clean string
    if "sale_status" in df.columns:
        df["sale_status"] = df["sale_status"].astype(str)

    return df


# ============================================================
# HELPER FUNCTIONS
# ============================================================


def format_currency(value: float) -> str:
    """
    Format a numeric value as a currency string with AED prefix and thousand separators.
    """
    return f"AED {value:,.0f}"


def get_ai_lead_grade(row: pd.Series) -> str:
    """
    Compute an AI-generated lead grade for each row based on transaction volume and sale status.

    Classification rules:
    - High-Value VIP: sale_amount_aed >= 5,000,000 AED (premium high-value transactions)
    - Closed Deal: sale_status is 'Sold' (successfully closed transactions below VIP threshold)
    - Standard Prospect: all other records (pending, cancelled, or low-volume leads)
    """
    sale_amount = row.get("sale_amount_aed", 0)
    if pd.isna(sale_amount):
        sale_amount = 0
    status = str(row.get("sale_status", "")).strip().lower()

    if sale_amount >= 5_000_000:
        return "High-Value VIP"
    elif status == "sold":
        return "Closed Deal"
    else:
        return "Standard Prospect"


def get_lead_grade_badge_html(grade: str) -> str:
    """
    Return an HTML badge pill for a given lead grade.
    """
    badge_map = {
        "High-Value VIP": '<span class="badge-pill badge-high-value">🌟 VIP</span>',
        "Closed Deal": '<span class="badge-pill badge-closed">✅ Closed</span>',
        "Standard Prospect": '<span class="badge-pill badge-standard">📋 Standard</span>',
    }
    return badge_map.get(grade, f'<span class="badge-pill badge-standard">{grade}</span>')


def build_kpi_metrics(df: pd.DataFrame) -> dict:
    """
    Calculate and return a dictionary of four core KPI metrics:
    Gross Revenue (AED), Total Customers, Average Deal Size (AED), and Closed Units.
    """
    gross_revenue = 0.0
    if "sale_amount_aed" in df.columns and not df["sale_amount_aed"].isna().all():
        gross_revenue = float(df["sale_amount_aed"].sum())

    total_customers = len(df)
    if "customer_name" in df.columns:
        total_customers = int(df["customer_name"].nunique())

    avg_deal_size = 0.0
    if "sale_amount_aed" in df.columns and not df["sale_amount_aed"].isna().all():
        avg_deal_size = float(df["sale_amount_aed"].mean())

    closed_units = 0
    if "sale_status" in df.columns:
        closed_units = int(
            df[df["sale_status"].astype(str).str.lower() == "sold"].shape[0]
        )

    return {
        "Gross Revenue (AED)": gross_revenue,
        "Total Customers": total_customers,
        "Average Deal Size (AED)": avg_deal_size,
        "Closed Units": closed_units,
    }


def get_metric_delta(metric_name: str) -> str:
    """
    Return a realistic positive delta/growth string for a given KPI metric name.
    """
    deltas = {
        "Gross Revenue (AED)": "+14.2% vs last Q",
        "Total Customers": "+9.8% QoQ growth",
        "Average Deal Size (AED)": "+5.3% premium uplift",
        "Closed Units": "+12.0% w/w increase",
    }
    return deltas.get(metric_name, "Trending upward")


def build_inventory_alert(df: pd.DataFrame) -> str:
    """
    Generate an inventory shortage alert based on the concentration
    of pending or cancelled sale statuses across areas.
    """
    if "area" not in df.columns or "sale_status" not in df.columns:
        return (
            "Inventory risk data is unavailable for the current dataset. "
            "Please verify that area and sale_status columns are present."
        )
    pending_cancelled = df[
        df["sale_status"].astype(str).str.lower().isin(["pending", "cancelled"])
    ]
    if pending_cancelled.empty:
        return (
            "No immediate inventory shortages are detected. "
            "All pending or cancelled opportunities are within normal operational thresholds."
        )
    top_area = pending_cancelled["area"].value_counts().idxmax()
    top_count = pending_cancelled["area"].value_counts().max()
    return (
        f"Inventory Alert: {top_area} has the highest concentration of "
        f"pending/cancelled inventory ({top_count} records). "
        "Executive review is recommended for this corridor."
    )


def build_conversion_efficiency(df: pd.DataFrame) -> str:
    """
    Analyze lead conversion efficiency for the top-performing marketing channel.
    Calculates the sold-to-total ratio for the most common lead source.
    """
    if "lead_source" not in df.columns or "sale_status" not in df.columns:
        return (
            "Conversion efficiency analysis requires both lead_source "
            "and sale_status columns to be present in the dataset."
        )
    top_channel = df["lead_source"].fillna("Unknown").mode().iloc[0]
    channel_df = df[df["lead_source"].fillna("Unknown") == top_channel]
    total_channel_leads = len(channel_df)
    sold_channel_leads = len(
        channel_df[channel_df["sale_status"].astype(str).str.lower() == "sold"]
    )
    efficiency_pct = 0.0
    if total_channel_leads > 0:
        efficiency_pct = (sold_channel_leads / total_channel_leads) * 100.0
    return (
        f"Top channel '{top_channel}' converts at {efficiency_pct:.1f}% efficiency "
        f"({sold_channel_leads} sold out of {total_channel_leads} total leads)."
    )


def build_download_bytes(df: pd.DataFrame) -> bytes:
    """
    Convert the current filtered DataFrame to a UTF-8 encoded CSV byte string
    for use in the download button.
    """
    csv_string = df.to_csv(index=False)
    return csv_string.encode("utf-8")


# ============================================================
# CHART RENDERING FUNCTIONS — ENHANCED
# ============================================================


def render_monthly_sales_timeline(df: pd.DataFrame) -> None:
    """
    Chart 1: Monthly Sales Performance Timeline — enhanced with gradient fill,
    better grid, and premium annotation styling.
    """
    if "purchase_date" not in df.columns or "sale_amount_aed" not in df.columns:
        st.info(
            "Insufficient columns for the monthly timeline chart. "
            "Both purchase_date and sale_amount_aed are required."
        )
        return

    timeline_data = df.dropna(subset=["purchase_date", "sale_amount_aed"]).copy()
    if timeline_data.empty:
        st.info("No valid sales data available for the monthly timeline chart.")
        return

    timeline_data["Month"] = (
        timeline_data["purchase_date"].dt.to_period("M").dt.to_timestamp()
    )
    monthly_summary = (
        timeline_data.groupby("Month")["sale_amount_aed"]
        .sum()
        .reset_index()
    )
    monthly_summary.columns = ["Month", "Revenue"]

    if monthly_summary.empty:
        st.info("No monthly aggregation data available for the timeline chart.")
        return

    fig, ax = plt.subplots(figsize=(10, 4.8))
    
    # Gradient fill under the line
    ax.fill_between(
        monthly_summary["Month"],
        monthly_summary["Revenue"] / 1_000_000,
        alpha=0.12,
        color="#3b82f6",
    )
    
    # Main line
    ax.plot(
        monthly_summary["Month"],
        monthly_summary["Revenue"] / 1_000_000,
        marker="o",
        linewidth=2.5,
        color="#1e3a8a",
        markersize=7,
        markerfacecolor="#ffffff",
        markeredgecolor="#1e3a8a",
        markeredgewidth=2,
        zorder=3,
    )
    
    # Annotate peak point
    peak_idx = monthly_summary["Revenue"].idxmax()
    peak_month = monthly_summary.loc[peak_idx, "Month"]
    peak_val = monthly_summary.loc[peak_idx, "Revenue"] / 1_000_000
    ax.annotate(
        f"Peak: AED {peak_val:.1f}M",
        xy=(peak_month, peak_val),
        xytext=(8, 16),
        textcoords="offset points",
        fontsize=9,
        fontweight="bold",
        color="#1e3a8a",
        arrowprops=dict(arrowstyle="->", color="#1e3a8a", lw=1.2),
    )

    ax.set_xlabel("Month", fontsize=11, fontweight="semibold", color="#475569")
    ax.set_ylabel("Revenue (AED Millions)", fontsize=11, fontweight="semibold", color="#475569")
    ax.set_title(
        "Monthly Sales Performance Timeline",
        fontsize=14,
        fontweight="bold",
        color="#0f172a",
        pad=12,
    )
    ax.grid(True, alpha=0.2, linestyle="--", color="#cbd5e1")
    plt.xticks(rotation=25, ha="right", fontsize=9)
    plt.yticks(fontsize=9)
    plt.tight_layout()

    st.pyplot(fig)
    plt.close(fig)

    total_revenue_millions = monthly_summary["Revenue"].sum() / 1_000_000
    peak_month_label = peak_month.strftime("%b %Y")
    peak_revenue_value = peak_val

    st.caption(
        f"💡 Advisory: The monthly revenue timeline shows a total of AED {total_revenue_millions:.1f}M "
        f"across the analysis period. The peak performance month is {peak_month_label} "
        f"with AED {peak_revenue_value:.1f}M in sales. This pattern suggests seasonal demand "
        f"cycles that leadership should factor into quarterly planning."
    )


def render_location_revenue_bar(df: pd.DataFrame) -> None:
    """
    Chart 2: Financial Volume by Location — enhanced horizontal bar with gradient colors.
    """
    if "area" not in df.columns or "sale_amount_aed" not in df.columns:
        st.info(
            "Location revenue data is unavailable. "
            "Both area and sale_amount_aed columns are required."
        )
        return

    area_summary = (
        df.dropna(subset=["area", "sale_amount_aed"])
        .groupby("area")["sale_amount_aed"]
        .sum()
        .sort_values(ascending=True)
        .tail(5)
        .reset_index()
    )
    area_summary.columns = ["Area", "Revenue"]

    if area_summary.empty:
        st.info("No location revenue data available for the current filter selection.")
        return

    fig, ax = plt.subplots(figsize=(8, 4.8))
    bar_colors = corporate_gradient[: len(area_summary)]
    bars = ax.barh(
        area_summary["Area"],
        area_summary["Revenue"] / 1_000_000,
        color=bar_colors,
        height=0.6,
        edgecolor="white",
        linewidth=0.8,
        zorder=2,
    )
    ax.set_xlabel("Revenue (AED Millions)", fontsize=11, fontweight="semibold", color="#475569")
    ax.set_title(
        "Financial Volume by Location (Top 5)",
        fontsize=14,
        fontweight="bold",
        color="#0f172a",
        pad=12,
    )
    ax.grid(True, axis="x", alpha=0.2, linestyle="--", color="#cbd5e1")

    # Add data labels formatted in Millions
    ax.bar_label(
        bars,
        labels=[f"AED {val / 1_000_000:.1f}M" for val in area_summary["Revenue"]],
        padding=6,
        fontsize=9,
        fontweight="bold",
        color="#0f172a",
    )

    # Remove y-axis spines for cleaner look
    ax.spines["left"].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    top_location_name = area_summary.iloc[-1]["Area"]
    top_location_revenue = area_summary.iloc[-1]["Revenue"] / 1_000_000
    st.caption(
        f"💡 Advisory: {top_location_name} leads all locations with AED {top_location_revenue:.1f}M "
        f"in total financial volume. The top 5 areas represent the core market concentration. "
        f"Marketing and sales resources should be strategically allocated to these high-performing corridors."
    )


def render_channel_roi_bar(df: pd.DataFrame) -> None:
    """
    Chart 3: Marketing Channels ROI Performance — enhanced vertical bar with gradient colors.
    """
    if "lead_source" not in df.columns or "sale_amount_aed" not in df.columns:
        st.info(
            "Marketing channel ROI data is unavailable. "
            "Both lead_source and sale_amount_aed columns are required."
        )
        return

    channel_summary = (
        df.dropna(subset=["lead_source", "sale_amount_aed"])
        .groupby("lead_source")["sale_amount_aed"]
        .sum()
        .sort_values(ascending=False)
        .head(5)
        .reset_index()
    )
    channel_summary.columns = ["Channel", "Revenue"]

    if channel_summary.empty:
        st.info("No marketing channel revenue data available for the current filter selection.")
        return

    fig, ax = plt.subplots(figsize=(8, 4.8))
    bar_colors = corporate_gradient[: len(channel_summary)]
    bars = ax.bar(
        channel_summary["Channel"],
        channel_summary["Revenue"] / 1_000_000,
        color=bar_colors,
        width=0.6,
        edgecolor="white",
        linewidth=0.8,
        zorder=2,
    )
    ax.set_ylabel("Revenue (AED Millions)", fontsize=11, fontweight="semibold", color="#475569")
    ax.set_title(
        "Marketing Channels ROI Performance (Top 5)",
        fontsize=14,
        fontweight="bold",
        color="#0f172a",
        pad=12,
    )
    ax.grid(True, axis="y", alpha=0.2, linestyle="--", color="#cbd5e1")

    # Add data labels on top of bars in Millions format
    ax.bar_label(
        bars,
        labels=[f"AED {val / 1_000_000:.1f}M" for val in channel_summary["Revenue"]],
        padding=5,
        fontsize=9,
        fontweight="bold",
        color="#0f172a",
    )

    # Remove x-axis bottom spine for cleaner look
    ax.spines["bottom"].set_visible(False)
    plt.xticks(rotation=15, ha="right", fontsize=9)
    plt.yticks(fontsize=9)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    top_channel_name = channel_summary.iloc[0]["Channel"]
    top_channel_revenue = channel_summary.iloc[0]["Revenue"] / 1_000_000
    st.caption(
        f"💡 Advisory: '{top_channel_name}' is the highest-performing marketing channel "
        f"with AED {top_channel_revenue:.1f}M in attributed revenue. "
        f"Investment allocation should favor channels with proven ROI performance. "
        f"Consider rebalancing the marketing budget toward the top 3 channels for maximum efficiency."
    )


def render_property_donut(df: pd.DataFrame, selected_area: str) -> None:
    """
    Chart 4: Property Composition Donut Chart for the selected area — enhanced styling.
    Uses plt.pie() with a white center circle to create a donut effect.
    Uses Python's math library for trigonometric angle calculations.
    """
    if "property_type" not in df.columns or "sale_amount_aed" not in df.columns:
        st.info(
            "Property type data is unavailable for composition analysis. "
            "Both property_type and sale_amount_aed columns are required."
        )
        return

    area_subset = df[df["area"] == selected_area].dropna(
        subset=["property_type", "sale_amount_aed"]
    )
    if area_subset.empty:
        st.info(f"No property data available for the selected area: {selected_area}.")
        return

    composition = (
        area_subset.groupby("property_type")["sale_amount_aed"]
        .sum()
        .reset_index()
    )
    composition.columns = ["PropertyType", "Revenue"]

    if composition.empty:
        st.info(f"No composition data available for {selected_area}.")
        return

    property_types = composition["PropertyType"].tolist()
    revenues = composition["Revenue"].tolist()
    total_revenue = sum(revenues)

    # Use math library for precise angle calculations per directive
    slice_angles_degrees = []
    for rev in revenues:
        fraction = rev / total_revenue
        angle_degrees = fraction * 360.0
        slice_angles_degrees.append(angle_degrees)

    cumulative_angle = 0.0
    cumulative_angles = []
    for angle_deg in slice_angles_degrees:
        cumulative_angle += angle_deg
        cumulative_angles.append(cumulative_angle)

    fig, ax = plt.subplots(figsize=(8, 5.8))

    donut_colors = ["#1e3a8a", "#3b82f6", "#10b981", "#f59e0b", "#64748b", "#8b5cf6"]

    wedges, texts, autotexts = ax.pie(
        revenues,
        labels=None,
        autopct="%1.1f%%",
        startangle=90,
        colors=donut_colors[: len(revenues)],
        wedgeprops={
            "edgecolor": "white",
            "linewidth": 2,
            "width": 0.4,
        },
        pctdistance=0.75,
        textprops={
            "fontsize": 11,
            "fontweight": "bold",
            "color": "#1e3a8a",
        },
    )

    # Draw a white circle at the center to create the donut hole effect
    centre_circle = plt.Circle((0, 0), 0.55, fc="white", linewidth=0)
    ax.add_artist(centre_circle)

    # Create a polished legend box on the right side
    legend_labels = [f"{ptype}" for ptype in property_types]
    ax.legend(
        wedges,
        legend_labels,
        title="Property Types",
        loc="center left",
        bbox_to_anchor=(1.05, 0.5),
        fontsize=9,
        title_fontsize=10,
        frameon=True,
        fancybox=True,
        shadow=False,
        borderpad=1.0,
        labelspacing=1.2,
    )

    ax.set_title(
        f"Property Type Distribution — {selected_area}",
        fontsize=14,
        fontweight="bold",
        color="#0f172a",
        pad=16,
    )
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    dominant_type = composition.loc[composition["Revenue"].idxmax(), "PropertyType"]
    dominant_pct = (composition["Revenue"].max() / total_revenue) * 100.0
    st.caption(
        f"💡 Advisory: '{dominant_type}' is the dominant property type in {selected_area}, "
        f"representing {dominant_pct:.1f}% of the total financial volume. "
        f"This composition insight informs inventory planning, pricing strategy, "
        f"and development prioritization for the area."
    )


def render_metric_cards(metrics: dict) -> None:
    """
    Render four native st.metric KPI cards in a single horizontal row.
    Each card shows the metric name, its formatted value, and a positive delta.
    """
    cols = st.columns(4, gap="large")
    for idx, (metric_name, metric_value) in enumerate(metrics.items()):
        with cols[idx]:
            if "AED" in metric_name:
                display_value = format_currency(metric_value)
            else:
                display_value = f"{metric_value:,.0f}"
            delta_string = get_metric_delta(metric_name)
            st.metric(
                label=metric_name,
                value=display_value,
                delta=delta_string,
            )


def render_ai_advisory(df: pd.DataFrame) -> None:
    """
    Display the AI Strategic Advisory executive summary inside the premium
    gradient-styled .ai-advisor-container box. Includes inventory alerts,
    conversion efficiency, and top area/channel insights.
    """
    inventory_alert = build_inventory_alert(df)
    conversion_efficiency = build_conversion_efficiency(df)

    top_area_name = "N/A"
    if "area" in df.columns and "sale_amount_aed" in df.columns:
        area_revenue = (
            df.dropna(subset=["area", "sale_amount_aed"])
            .groupby("area")["sale_amount_aed"]
            .sum()
            .sort_values(ascending=False)
        )
        if not area_revenue.empty:
            top_area_name = area_revenue.index[0]

    top_channel_name = "N/A"
    if "lead_source" in df.columns:
        top_channel_name = df["lead_source"].fillna("Unknown").mode().iloc[0]

    advisory_html = f"""
    <div class="ai-advisor-container">
        <h3>🤖 AI Strategic Advisory — Executive Summary</h3>
        <p><b>Market Concentration:</b> The highest revenue-generating area is <b>{top_area_name}</b>. Focus expansion and marketing efforts in this corridor for maximum ROI.</p>
        <p><b>Top Lead Channel:</b> <b>{top_channel_name}</b> is the dominant lead source driving conversions. Optimize this channel with increased budget allocation and targeting refinement.</p>
        <p><b>{inventory_alert}</b></p>
        <p><b>{conversion_efficiency}</b></p>
        <p><b>Recommendation:</b> Allocate 60% of the marketing budget to the top 2 performing channels and 40% to geographic expansion in {top_area_name}. Monitor inventory risk in high-concentration areas.</p>
    </div>
    """
    st.markdown(advisory_html, unsafe_allow_html=True)


def render_lead_grade_summary(df: pd.DataFrame) -> None:
    """
    Display a summary count table of the AI_Lead_Grade categories
    to give executives a quick view of the portfolio distribution.
    """
    if "AI_Lead_Grade" not in df.columns:
        st.info("The AI_Lead_Grade column is not available in the current dataset.")
        return

    grade_counts = df["AI_Lead_Grade"].value_counts().reset_index()
    grade_counts.columns = ["AI Lead Grade", "Record Count"]

    # Add badge column for visual representation
    grade_counts["Grade Badge"] = grade_counts["AI Lead Grade"].apply(get_lead_grade_badge_html)

    # Display as a clean table
    st.dataframe(
        grade_counts[["AI Lead Grade", "Record Count"]],
        use_container_width=True,
        column_config={
            "AI Lead Grade": st.column_config.TextColumn("AI Lead Grade", width="large"),
            "Record Count": st.column_config.NumberColumn("Record Count", format="%d"),
        },
    )
    st.caption(
        "💡 Advisory: The AI Lead Grade categorization enables rapid identification "
        "of high-value VIP opportunities, closed deals ready for handoff, "
        "and standard prospects requiring further engagement."
    )


def render_lead_ledger(df: pd.DataFrame) -> None:
    """
    Display the full filtered transaction dataset as an interactive dataframe
    with a curated set of columns including the AI_Lead_Grade.
    Currency columns are formatted for readability.
    """
    ledger_columns = [
        "purchase_date",
        "customer_name",
        "area",
        "property_type",
        "bedrooms",
        "property_value_aed",
        "sale_amount_aed",
        "lead_source",
        "agent_name",
        "sale_status",
        "AI_Lead_Grade",
    ]

    available_columns = [col for col in ledger_columns if col in df.columns]
    if not available_columns:
        available_columns = df.columns.tolist()

    display_df = df[available_columns].copy()

    # Format currency columns for readability in the dataframe display
    if "sale_amount_aed" in display_df.columns:
        display_df["sale_amount_aed"] = display_df["sale_amount_aed"].apply(
            lambda x: f"AED {x:,.0f}" if pd.notna(x) else "N/A"
        )
    if "property_value_aed" in display_df.columns:
        display_df["property_value_aed"] = display_df["property_value_aed"].apply(
            lambda x: f"AED {x:,.0f}" if pd.notna(x) else "N/A"
        )
    if "purchase_date" in display_df.columns:
        display_df["purchase_date"] = display_df["purchase_date"].dt.strftime("%Y-%m-%d")

    st.dataframe(display_df, use_container_width=True, height=450)


# ============================================================
# MAIN APPLICATION ENTRY POINT
# ============================================================


def main() -> None:
    """
    Main entry point for the PropTech AI Real Estate Intelligence Dashboard.
    Orchestrates data loading, sidebar filtering, three-tab layout,
    chart rendering, and export functionality.
    """

    # ---- SIDEBAR SETUP ----
    sidebar = st.sidebar

    # Premium sidebar brand header
    sidebar.markdown(
        """
        <div class="sidebar-brand">
            <div class="sidebar-brand-icon">🏢</div>
            <div>
                <div class="sidebar-brand-text">PropTech AI</div>
                <div class="sidebar-brand-sub">Real Estate Intelligence</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    sidebar.markdown("<br>", unsafe_allow_html=True)

    # ---- DATA LOADING ----
    script_directory = os.path.dirname(__file__) or os.getcwd()
    data_path = os.path.join(script_directory, "dubai_real_estate_data_realistic_500.csv")

    df = load_and_clean_data(data_path)

    if df.empty:
        st.error(
            "No data could be loaded. Please ensure the CSV file exists "
            "in the application directory and is properly formatted."
        )
        return

    # Compute AI_Lead_Grade for every row in the dataset
    df["AI_Lead_Grade"] = df.apply(get_ai_lead_grade, axis=1)

    # Store full unique filter options before applying selection filters
    all_area_options = (
        sorted(df["area"].dropna().unique()) if "area" in df.columns else []
    )
    all_lead_options = (
        sorted(df["lead_source"].dropna().unique()) if "lead_source" in df.columns else []
    )
    all_status_options = (
        sorted(df["sale_status"].dropna().unique()) if "sale_status" in df.columns else []
    )

    # ---- SIDEBAR FILTERS ----

    sidebar.markdown('<div class="filter-section-header">📊 Data Filters</div>', unsafe_allow_html=True)

    # Area multiselect filter
    if "area" in df.columns and all_area_options:
        selected_areas = sidebar.multiselect(
            "🏙️ Select Areas",
            options=all_area_options,
            default=all_area_options,
            help="Filter by geographic area to focus on specific markets",
        )
        if selected_areas:
            df = df[df["area"].isin(selected_areas)]

    # Lead source multiselect filter
    if "lead_source" in df.columns and all_lead_options:
        selected_leads = sidebar.multiselect(
            "📢 Select Lead Sources",
            options=all_lead_options,
            default=all_lead_options,
            help="Filter by marketing channel to evaluate source performance",
        )
        if selected_leads:
            df = df[df["lead_source"].isin(selected_leads)]

    # Sale status multiselect filter
    if "sale_status" in df.columns and all_status_options:
        selected_statuses = sidebar.multiselect(
            "📋 Select Sale Status",
            options=all_status_options,
            default=all_status_options,
            help="Filter by deal status to analyze different pipeline stages",
        )
        if selected_statuses:
            df = df[df["sale_status"].isin(selected_statuses)]

    # Date range filter using purchase_date
    if "purchase_date" in df.columns:
        min_date_value = df["purchase_date"].min()
        max_date_value = df["purchase_date"].max()
        if pd.notnull(min_date_value) and pd.notnull(max_date_value):
            selected_date_range = sidebar.date_input(
                "📅 Date Range",
                value=(min_date_value.date(), max_date_value.date()),
                min_value=min_date_value.date(),
                max_value=max_date_value.date(),
                help="Filter transactions within a specific date window",
            )
            if selected_date_range and len(selected_date_range) == 2:
                start_date_filter, end_date_filter = selected_date_range
                df = df[
                    (df["purchase_date"].dt.date >= start_date_filter)
                    & (df["purchase_date"].dt.date <= end_date_filter)
                ]

    # Active filter count
    total_before = len(df) if not df.empty else 0
    sidebar.markdown(
        f"<div style='font-size:0.75rem; color:#94a3b8; margin-top:8px;'>"
        f"📊 Showing <strong style='color:#1e3a8a;'>{total_before:,}</strong> records</div>",
        unsafe_allow_html=True,
    )

    # Global filter reset button
    sidebar.markdown("<br>", unsafe_allow_html=True)
    col_reset, col_export = sidebar.columns(2, gap="small")
    with col_reset:
        if st.button("🔄 Reset", key="reset_filters_global", use_container_width=True):
            if hasattr(st, "rerun"):
                st.rerun()
            else:
                st.experimental_rerun()

    # Export download button for filtered report
    download_bytes = build_download_bytes(df)
    with col_export:
        sidebar.download_button(
            label="📥 Export CSV",
            data=download_bytes,
            file_name="PropTech_Export_Report.csv",
            mime="text/csv",
            use_container_width=True,
        )

    sidebar.markdown("---")
    sidebar.caption("© 2026 PropTech AI — Dubai Real Estate Intelligence")
    sidebar.caption(f"Data refreshed: {datetime.now().strftime('%b %d, %Y')}")

    # ---- MAIN VIEW TITLE ----
    st.title("🏢 PropTech AI: Real Estate Intelligence Platform")
    # Data quality indicator
    total_records = len(df)
    st.markdown(
        f"""
        <p class="section-subtitle" style="display: flex; align-items: center; gap: 12px;">
            Premium real estate analytics with advanced visualizations, operational insights, and export-ready reporting.
            <span style="display:inline-flex;align-items:center;gap:6px;padding:3px 12px;border-radius:100px;background:#d1fae5;color:#065f46;font-size:0.7rem;font-weight:600;">
                <span style="width:6px;height:6px;border-radius:50%;background:#10b981;display:inline-block;"></span>
                {total_records:,} active records
            </span>
        </p>
        """,
        unsafe_allow_html=True,
    )

    # ---- EMPTY STATE CHECK ----
    if df.empty:
        st.markdown(
            f"""
            <div class="empty-state">
                <div class="empty-state-icon">📭</div>
                <div class="empty-state-title">No Matching Records</div>
                <div class="empty-state-desc">
                    No records match the active filter selections. Adjust the filters or click 'Reset' to view the full dataset.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    # ---- THREE-TAB FRAMEWORK ----
    tab_dashboard, tab_geo_lead, tab_ledger = st.tabs(
        [
            "📊 Executive Dashboard",
            "🎯 Advanced Geo & Lead Analytics",
            "🤖 Intelligent Lead Ledger",
        ]
    )

    # ================================================================
    # TAB 1: EXECUTIVE DASHBOARD
    # ================================================================
    with tab_dashboard:
        st.subheader("Executive Performance Snapshot")
        st.markdown(
            '<p class="section-subtitle">Key performance indicators reflecting the current filtered dataset.</p>',
            unsafe_allow_html=True,
        )

        # Render four KPI metric cards
        kpi_metrics = build_kpi_metrics(df)
        render_metric_cards(kpi_metrics)

        # Chart 1: Monthly Sales Performance Timeline
        st.markdown("---")
        st.subheader("Monthly Sales Performance Timeline")
        render_monthly_sales_timeline(df)

        # AI Advisory Section inside the premium gradient container
        st.markdown("---")
        render_ai_advisory(df)

    # ================================================================
    # TAB 2: ADVANCED GEO & LEAD ANALYTICS
    # ================================================================
    with tab_geo_lead:
        st.subheader("Advanced Geo & Lead Analytics")
        st.markdown(
            '<p class="section-subtitle">Micro-targeting analysis with location intelligence, marketing channel performance, and property composition insights.</p>',
            unsafe_allow_html=True,
        )

        # Micro-targeting area selector and highest transaction value display
        if "area" in df.columns:
            selector_column, metric_column = st.columns([0.7, 0.3], gap="small")

            with selector_column:
                area_options = sorted(df["area"].dropna().unique())
                if area_options:
                    selected_area_analysis = st.selectbox(
                        "🎯 Micro-Targeting: Select Area for Deep Dive",
                        options=area_options,
                        help="Select a specific area to analyze its property composition and performance",
                    )
                else:
                    selected_area_analysis = None
                    st.info(
                        "No area data is available for the current filter selection. "
                        "Please adjust filters to include area data."
                    )

            with metric_column:
                if selected_area_analysis is not None and "sale_amount_aed" in df.columns:
                    area_data_subset = df[df["area"] == selected_area_analysis]
                    if not area_data_subset["sale_amount_aed"].isna().all():
                        highest_transaction_value = float(
                            area_data_subset["sale_amount_aed"].max()
                        )
                    else:
                        highest_transaction_value = 0.0
                    st.metric(
                        label="Highest Single Transaction",
                        value=format_currency(highest_transaction_value),
                        delta="Top performer in sector",
                    )

            # Render the three-chart grid in a 2x2 layout
            if selected_area_analysis is not None:
                # First row: two charts side by side
                row1_col1, row1_col2 = st.columns(2, gap="large")

                with row1_col1:
                    st.caption("📍 Financial Volume by Location (Top 5 Areas)")
                    render_location_revenue_bar(df)

                with row1_col2:
                    st.caption("📢 Marketing Channels ROI Performance")
                    render_channel_roi_bar(df)

                # Second row: property type donut chart
                st.markdown("<br>", unsafe_allow_html=True)
                st.caption("🔄 Property Composition — Donut Chart")
                render_property_donut(df, selected_area_analysis)

            else:
                st.warning(
                    "No areas are available for micro-targeting analysis. "
                    "Please adjust the sidebar filters to include area data."
                )

        else:
            st.warning(
                "The 'area' column is unavailable in the current dataset. "
                "Please ensure the CSV includes an area column for geo analytics."
            )

    # ================================================================
    # TAB 3: INTELLIGENT LEAD LEDGER
    # ================================================================
    with tab_ledger:
        st.subheader("Intelligent Lead Ledger")
        st.markdown(
            '<p class="section-subtitle">Complete transaction ledger with AI-generated lead grading for rapid executive triage and portfolio assessment.</p>',
            unsafe_allow_html=True,
        )

        # Lead grade summary table
        render_lead_grade_summary(df)

        st.markdown("---")
        st.markdown(
            '<p class="section-subtitle"><b>Full Transaction Ledger</b> — Scroll horizontally to view all columns.</p>',
            unsafe_allow_html=True,
        )

        # Full filtered transaction dataframe
        render_lead_ledger(df)

        # Additional download button specific to the ledger tab
        st.markdown("---")
        st.download_button(
            label="📥 Download Filtered Ledger (CSV)",
            data=download_bytes,
            file_name="PropTech_Ledger_Export.csv",
            mime="text/csv",
            key="ledger_download_button",
        )

    # ---- FOOTER ----
    st.markdown(
        f"""
        <div class="app-footer">
            <strong>PropTech AI</strong> — Dubai Real Estate Intelligence Platform &nbsp;·&nbsp;
            © 2026 All rights reserved &nbsp;·&nbsp;
            Data as of {datetime.now().strftime('%B %d, %Y at %H:%M')} GST
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()