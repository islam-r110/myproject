import pandas as pd
import streamlit as st
from pathlib import Path

st.set_page_config(page_title="Dubai Real Estate Dashboard", layout="wide")

DATA_PATH = Path(__file__).resolve().parent / "data" / "dubai_real_estate_data_realistic_500.csv"


def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        st.error(f"CSV file not found:\n{path}")
        return pd.DataFrame()

    try:
        df = pd.read_csv(path)
        return df
    except Exception as e:
        st.error(f"Failed to read CSV file: {e}")
        return pd.DataFrame()


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    df = df.drop_duplicates().copy()

    # Clean text columns
    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = df[col].astype(str).str.strip()

    # Convert numeric columns if they exist
    numeric_columns = [
        "budget_aed",
        "property_value_aed",
        "sale_amount_aed",
        "customer_age",
        "bedrooms",
    ]
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Convert date if it exists
    if "purchase_date" in df.columns:
        df["purchase_date"] = pd.to_datetime(df["purchase_date"], errors="coerce")

    # Create month column for charts
    if "purchase_date" in df.columns:
        df["sale_month"] = df["purchase_date"].dt.to_period("M").astype(str)

    return df


def main():
    st.title("Dubai Real Estate Dashboard")

    df = load_data(DATA_PATH)
    if df.empty:
        return

    df = clean_data(df)

    required_columns = [
        "area",
        "nationality",
        "lead_source",
        "property_type",
        "agent_name",
        "sale_amount_aed",
    ]

    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        st.error(f"Missing required columns in the CSV: {missing}")
        return

    st.sidebar.header("Filters")

    area_options = sorted([str(x) for x in df["area"].dropna().astype(str).unique()])
    nationality_options = sorted(
        [str(x) for x in df["nationality"].dropna().astype(str).unique()]
    )

    area_filter = st.sidebar.multiselect("Select Area", options=area_options)
    nationality_filter = st.sidebar.multiselect(
        "Select Nationality", options=nationality_options
    )

    if area_filter:
        df = df[df["area"].astype(str).isin(area_filter)]

    if nationality_filter:
        df = df[df["nationality"].astype(str).isin(nationality_filter)]

    if df.empty:
        st.warning("No data matches the selected filters.")
        return

    st.header("Top Selling Areas")
    area_sales = df.groupby("area")["sale_amount_aed"].sum().sort_values(ascending=False)
    st.bar_chart(area_sales)

    st.header("Top Buying Nationalities")
    nationality_sales = df.groupby("nationality")["sale_amount_aed"].sum().sort_values(ascending=False)
    st.bar_chart(nationality_sales)

    st.header("Sales by Lead Source")
    source_sales = df.groupby("lead_source")["sale_amount_aed"].sum().sort_values(ascending=False)
    st.bar_chart(source_sales)

    st.header("Sales by Property Type")
    property_sales = df.groupby("property_type")["sale_amount_aed"].sum().sort_values(ascending=False)
    st.bar_chart(property_sales)

    st.header("Top Agents")
    agent_sales = df.groupby("agent_name")["sale_amount_aed"].sum().sort_values(ascending=False)
    st.write(agent_sales.head(10))

    st.header("Monthly Sales")
    monthly_sales = df.groupby("sale_month")["sale_amount_aed"].sum().sort_index()
    st.line_chart(monthly_sales)


if __name__ == "__main__":
    main()