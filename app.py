import io
import zipfile
import requests
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------
st.set_page_config(
    page_title="Mystery Retail Intelligence",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Mystery Retail Intelligence")
st.markdown(
    "### ALGOTHON'26 — ALG-DATA-01: The Mystery Dataset"
)

st.write(
    "An end-to-end data science dashboard for discovering patterns, "
    "customer behavior, anomalies and revenue trends from the UCI Online Retail dataset."
)


# ---------------------------------------------------------
# DATA DOWNLOAD
# ---------------------------------------------------------
DATA_URL = "https://archive.ics.uci.edu/static/public/352/online+retail.zip"


@st.cache_data
def load_data():

    response = requests.get(DATA_URL, timeout=60)
    response.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(response.content)) as z:

        excel_file = [
            name for name in z.namelist()
            if name.lower().endswith(".xlsx")
        ][0]

        with z.open(excel_file) as file:
            df = pd.read_excel(file)

    return df


# ---------------------------------------------------------
# DATA CLEANING
# ---------------------------------------------------------
@st.cache_data
def clean_data(df):

    data = df.copy()

    data["InvoiceDate"] = pd.to_datetime(
        data["InvoiceDate"],
        errors="coerce"
    )

    data["IsCancellation"] = (
        data["InvoiceNo"]
        .astype(str)
        .str.startswith("C")
    )

    original_rows = len(data)

    data = data.drop_duplicates()

    data = data.dropna(
        subset=["InvoiceDate", "Description"]
    )

    data = data[~data["IsCancellation"]]

    data = data[
        (data["Quantity"] > 0) &
        (data["UnitPrice"] > 0)
    ]

    data["Amount"] = (
        data["Quantity"] *
        data["UnitPrice"]
    )

    removed_rows = original_rows - len(data)

    return data, original_rows, removed_rows


# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------
with st.spinner("Downloading and preparing dataset..."):

    try:
        raw_df = load_data()
        df, original_rows, removed_rows = clean_data(raw_df)

    except Exception as e:

        st.error("Unable to load the dataset.")
        st.exception(e)
        st.stop()


# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------
st.sidebar.title("🔎 Dashboard")

st.sidebar.success(
    "Dataset loaded successfully!"
)

st.sidebar.write(
    f"Original rows: {original_rows:,}"
)

st.sidebar.write(
    f"Clean rows: {len(df):,}"
)

st.sidebar.write(
    f"Removed rows: {removed_rows:,}"
)


# ---------------------------------------------------------
# KPI SECTION
# ---------------------------------------------------------
total_revenue = df["Amount"].sum()
total_orders = df["InvoiceNo"].nunique()
total_customers = df["CustomerID"].nunique()
total_products = df["StockCode"].nunique()
total_countries = df["Country"].nunique()

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric(
    "💰 Revenue",
    f"£{total_revenue:,.0f}"
)

col2.metric(
    "🧾 Orders",
    f"{total_orders:,}"
)

col3.metric(
    "👥 Customers",
    f"{total_customers:,}"
)

col4.metric(
    "📦 Products",
    f"{total_products:,}"
)

col5.metric(
    "🌍 Countries",
    f"{total_countries:,}"
)


st.divider()


# ---------------------------------------------------------
# TABS
# ---------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "📈 Discovery",
        "👥 Customers",
        "🚨 Anomalies",
        "🔮 Prediction",
        "💡 Hypotheses"
    ]
)


# =========================================================
# TAB 1 — DISCOVERY
# =========================================================
with tab1:

    st.header("📈 Data Discovery")

    st.write(
        "Explore revenue patterns, top products and geographic performance."
    )

    # Daily revenue
    daily_sales = (
        df.groupby(
            df["InvoiceDate"].dt.date
        )["Amount"]
        .sum()
        .reset_index()
    )

    daily_sales.columns = [
        "Date",
        "Revenue"
    ]

    fig = px.line(
        daily_sales,
        x="Date",
        y="Revenue",
        title="Daily Revenue Trend"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # Top products
    st.subheader("🏆 Top 10 Products")

    top_products = (
        df.groupby("Description")["Amount"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
    )

    fig_products = px.bar(
        top_products.sort_values("Amount"),
        x="Amount",
        y="Description",
        orientation="h",
        title="Top 10 Products by Revenue"
    )

    st.plotly_chart(
        fig_products,
        use_container_width=True
    )

    # Countries
    st.subheader("🌍 Top Countries")

    country_sales = (
        df.groupby("Country")["Amount"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
    )

    fig_country = px.bar(
        country_sales.sort_values("Amount"),
        x="Amount",
        y="Country",
        orientation="h",
        title="Top 10 Countries by Revenue"
    )

    st.plotly_chart(
        fig_country,
        use_container_width=True
    )


# =========================================================
# TAB 2 — CUSTOMERS
# =========================================================
with tab2:

    st.header("👥 Customer Intelligence")

    customer_data = df.dropna(
        subset=["CustomerID"]
    ).copy()

    if len(customer_data) == 0:

        st.warning(
            "CustomerID information is not available."
        )

    else:

        snapshot_date = (
            customer_data["InvoiceDate"].max()
            + pd.Timedelta(days=1)
        )

        rfm = (
            customer_data
            .groupby("CustomerID")
            .agg(
                Recency=(
                    "InvoiceDate",
                    lambda x:
                    (snapshot_date - x.max()).days
                ),
                Frequency=(
                    "InvoiceNo",
                    "nunique"
                ),
                Monetary=(
                    "Amount",
                    "sum"
                )
            )
            .reset_index()
        )

        # Ranking avoids qcut duplicate-edge errors
        rfm["R"] = pd.qcut(
            rfm["Recency"].rank(method="first"),
            4,
            labels=[4, 3, 2, 1]
        ).astype(int)

        rfm["F"] = pd.qcut(
            rfm["Frequency"].rank(method="first"),
            4,
            labels=[1, 2, 3, 4]
        ).astype(int)

        rfm["M"] = pd.qcut(
            rfm["Monetary"].rank(method="first"),
            4,
            labels=[1, 2, 3, 4]
        ).astype(int)

        def segment(row):

            if (
                row["R"] >= 3
                and row["F"] >= 3
                and row["M"] >= 3
            ):
                return "Champions"

            elif (
                row["R"] >= 3
                and row["F"] >= 2
            ):
                return "Loyal / Active"

            elif (
                row["R"] <= 2
                and row["F"] >= 3
            ):
                return "At Risk"

            elif (
                row["R"] <= 2
                and row["F"] <= 2
            ):
                return "Hibernating"

            else:
                return "Potential"

        rfm["Segment"] = rfm.apply(
            segment,
            axis=1
        )

        segment_counts = (
            rfm["Segment"]
            .value_counts()
            .reset_index()
        )

        segment_counts.columns = [
            "Segment",
            "Customers"
        ]

        fig_segment = px.bar(
            segment_counts,
            x="Segment",
            y="Customers",
            title="Customer Segmentation"
        )

        st.plotly_chart(
            fig_segment,
            use_container_width=True
        )

        st.subheader("Customer RFM Table")

        st.dataframe(
            rfm.sort_values(
                "Monetary",
                ascending=False
            ).head(20),
            use_container_width=True
        )


# =========================================================
# TAB 3 — ANOMALIES
# =========================================================
with tab3:

    st.header("🚨 Anomaly Detection")

    st.write(
        "Isolation Forest identifies unusual transaction patterns."
    )

    anomaly_data = df[
        [
            "Quantity",
            "UnitPrice",
            "Amount"
        ]
    ].copy()

    anomaly_data["LogQuantity"] = np.log1p(
        anomaly_data["Quantity"]
    )

    anomaly_data["LogAmount"] = np.log1p(
        anomaly_data["Amount"]
    )

    model = IsolationForest(
        n_estimators=200,
        contamination=0.01,
        random_state=42
    )

    X = anomaly_data[
        [
            "LogQuantity",
            "UnitPrice",
            "LogAmount"
        ]
    ]

    anomaly_data["Prediction"] = (
        model.fit_predict(X)
    )

    anomalies = anomaly_data[
        anomaly_data["Prediction"] == -1
    ]

    st.metric(
        "Detected Anomalies",
        f"{len(anomalies):,}"
    )

    st.warning(
        "An anomaly is a statistically unusual transaction. "
        "It does not automatically mean fraud or an error."
    )

    st.subheader(
        "Highest-Value Anomalous Transactions"
    )

    st.dataframe(
        anomalies
        .sort_values(
            "Amount",
            ascending=False
        )
        .head(30),
        use_container_width=True
    )


# =========================================================
# TAB 4 — PREDICTION
# =========================================================
with tab4:

    st.header("🔮 Revenue Prediction")

    st.write(
        "A Random Forest model predicts revenue using historical "
        "revenue patterns and calendar features."
    )

    daily = (
        df.groupby(
            df["InvoiceDate"].dt.floor("D")
        )
        .agg(
            Revenue=("Amount", "sum"),
            Orders=("InvoiceNo", "nunique")
        )
        .reset_index()
    )

    daily.columns = [
        "Date",
        "Revenue",
        "Orders"
    ]

    daily = daily.sort_values("Date")

    daily["lag1"] = (
        daily["Revenue"].shift(1)
    )

    daily["lag7"] = (
        daily["Revenue"].shift(7)
    )

    daily["rolling7"] = (
        daily["Revenue"]
        .shift(1)
        .rolling(7)
        .mean()
    )

    daily["dayofweek"] = (
        daily["Date"].dt.dayofweek
    )

    daily["month"] = (
        daily["Date"].dt.month
    )

    daily = daily.dropna()

    features = [
        "lag1",
        "lag7",
        "rolling7",
        "dayofweek",
        "month"
    ]

    X = daily[features]
    y = daily["Revenue"]

    split = int(len(daily) * 0.80)

    X_train = X.iloc[:split]
    X_test = X.iloc[split:]

    y_train = y.iloc[:split]
    y_test = y.iloc[split:]

    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )

    model.fit(
        X_train,
        y_train
    )

    predictions = model.predict(
        X_test
    )

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    rmse = mean_squared_error(
        y_test,
        predictions
    ) ** 0.5

    col1, col2 = st.columns(2)

    col1.metric(
        "MAE",
        f"£{mae:,.2f}"
    )

    col2.metric(
        "RMSE",
        f"£{rmse:,.2f}"
    )

    results = pd.DataFrame({
        "Date": daily.iloc[split:]["Date"],
        "Actual": y_test.values,
        "Predicted": predictions
    })

    fig_prediction = px.line(
        results,
        x="Date",
        y=["Actual", "Predicted"],
        title="Actual vs Predicted Revenue"
    )

    st.plotly_chart(
        fig_prediction,
        use_container_width=True
    )

    st.info(
        "The chronological train/test split reduces data leakage. "
        "This forecast is a prototype and should not be treated as a "
        "production financial forecast."
    )


# =========================================================
# TAB 5 — HYPOTHESES
# =========================================================
with tab5:

    st.header("💡 Hypothesis Testing")

    st.subheader(
        "Hypothesis: Higher quantity orders may have different pricing behavior."
    )

    correlation = (
        df[
            [
                "Quantity",
                "UnitPrice"
            ]
        ]
        .corr(method="spearman")
        .iloc[0, 1]
    )

    st.metric(
        "Spearman Correlation: Quantity vs Unit Price",
        round(correlation, 3)
    )

    quantity_cutoff = (
        df["Quantity"].quantile(0.90)
    )

    high_quantity = df[
        df["Quantity"] >= quantity_cutoff
    ]

    normal_quantity = df[
        df["Quantity"] < quantity_cutoff
    ]

    high_median = (
        high_quantity["UnitPrice"].median()
    )

    normal_median = (
        normal_quantity["UnitPrice"].median()
    )

    col1, col2 = st.columns(2)

    col1.metric(
        "Top 10% Quantity Median Price",
        f"£{high_median:.2f}"
    )

    col2.metric(
        "Remaining Transactions Median Price",
        f"£{normal_median:.2f}"
    )

    st.write(
        "The analysis helps investigate whether large-quantity "
        "transactions are associated with different unit-price behavior."
    )

    st.subheader("Relationship Matrix")

    correlation_matrix = df[
        [
            "Quantity",
            "UnitPrice",
            "Amount"
        ]
    ].corr(method="spearman")

    fig_corr = px.imshow(
        correlation_matrix,
        text_auto=True,
        title="Spearman Relationship Matrix"
    )

    st.plotly_chart(
        fig_corr,
        use_container_width=True
    )


# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------
st.divider()

st.caption(
    "ALGOTHON'26 | Mystery Retail Intelligence | "
    "Dataset: UCI Online Retail | "
    "Statistical anomalies are not automatically fraud."
)
