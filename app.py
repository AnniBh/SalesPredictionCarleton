import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from prophet import Prophet
from prophet.plot import plot_components_plotly
from datetime import datetime
import numpy as np
import time as time
import io
from fpdf import FPDF
import os
import calendar

# Set page configuration to wide mode
st.set_page_config(layout="wide", page_title="Sales Dashboard Chaitime", page_icon="📊")

# Load the data
data = pd.read_csv("Updated_Sales_Data.csv")

# Convert the "Date" column to datetime format
data["Date"] = pd.to_datetime(data["Date"])
data["Year"] = data["Date"].dt.year
data["Year-Month"] = data["Date"].dt.to_period("M")
data["Month"] = data["Date"].dt.month  # Ensure this column is created
data["Month Name"] = data["Month"].apply(lambda x: calendar.month_name[x])

# Initialize session state for charts
if "dashboard_charts" not in st.session_state:
    st.session_state.dashboard_charts = []

if "prediction_charts" not in st.session_state:
    st.session_state.prediction_charts = []

def save_chart_to_file(fig, filename):
    """Save a Matplotlib chart as an image."""
    file_path = os.path.join(os.getcwd(), filename)  # Use absolute path
    print("Saving chart to:", file_path)
    fig.savefig(file_path, bbox_inches="tight")
    plt.close(fig)
    return file_path  # Return the full path for use later

def save_prophet_plot_to_file(fig, filename):
    """Save a Prophet-generated plot as an image."""
    file_path = os.path.join(os.getcwd(), filename)  # Use absolute path
    print("Saving Prophet plot to:", file_path)
    fig.savefig(file_path, bbox_inches="tight")
    plt.close(fig)
    return file_path  # Return the full path

def generate_pdf_report(dashboard_figs, prediction_figs, output_file="report.pdf"):
    print("generate_pdf_report")
    print("dashboard_figs", dashboard_figs)
    print("prediction_figs", prediction_figs)
    pdf = FPDF()
    pdf.add_page()

    # Dashboard Section
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "Dashboard Summary", ln=1)
    pdf.ln(10)

    for chart_path in dashboard_figs:
        if os.path.exists(chart_path):  # Check if the file exists
            print("prediction_figs", chart_path)
            pdf.image(chart_path, x=10, y=None, w=190)
            pdf.ln(10)
        else:
            pdf.set_font("Arial", "I", 12)
            pdf.cell(0, 10, f"Missing chart: {chart_path}", ln=1)
            pdf.ln(10)

    # Predictions Section
    pdf.add_page()
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "Predictions Summary", ln=1)
    pdf.ln(10)

    for chart_path in prediction_figs:
        if os.path.exists(chart_path):  # Check if the file exists
            print("prediction_figs", chart_path)
            pdf.image(chart_path, x=10, y=None, w=190)
            pdf.ln(10)
        else:
            pdf.set_font("Arial", "I", 12)
            pdf.cell(0, 10, f"Missing chart: {chart_path}", ln=1)
            pdf.ln(10)

    pdf.output(output_file)
    return output_file

# Sidebar for Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Dashboard", "Enhanced Predictions", "Generate Report"])

if page == "Dashboard":
    # Main Dashboard Logic
    st.title("Sales Dashboard for Chaitime Carleton University Store")

    # Sidebar Filters
    st.sidebar.header("Filters")

    # Year filter
    current_year = datetime.now().year
    year_filter = st.sidebar.selectbox(
        "Select Year:",
        options=sorted(data["Year"].unique(), reverse=True),
        index=sorted(data["Year"].unique(), reverse=True).index(current_year)
    )

    # Filter data by selected year
    filtered_data_by_year = data[data["Year"] == year_filter]

    # Category filter
    category_filter = st.sidebar.multiselect(
        "Select Categories:",
        options=filtered_data_by_year["Category"].unique(),
        default=filtered_data_by_year["Category"].unique()
    )

    # Item filter
    item_filter = st.sidebar.multiselect(
        "Select Items:",
        options=filtered_data_by_year[filtered_data_by_year["Category"].isin(category_filter)]["Item Name"].unique(),
        default=filtered_data_by_year[filtered_data_by_year["Category"].isin(category_filter)]["Item Name"].unique()
    )

    # Apply all filters
    filtered_data = filtered_data_by_year[
        (filtered_data_by_year["Category"].isin(category_filter)) &
        (filtered_data_by_year["Item Name"].isin(item_filter))
    ]
    tooltip = "Report will be generated for the current applied filters"
    # Create a placeholder for the spinner
    spinner_placeholder = st.sidebar.empty()

    if st.sidebar.button("Export Data", help=tooltip):
        with spinner_placeholder:
            with st.spinner("Preparing your report..."):  # Spinner will now appear in the sidebar
                time.sleep(1)  # Simulate processing delay
                
                # Generate the CSV file
                export_file = io.BytesIO()
                filtered_data.to_csv(export_file, index=False)
                export_file.seek(0)
                st.sidebar.success("Your file is ready for download!")
                st.sidebar.download_button(
                    label="Download CSV",
                    data=export_file,
                    file_name=f"Sales_Report_{year_filter}.csv",
                    mime="text/csv"
                )

    # Clear the spinner placeholder once done
    spinner_placeholder.empty()

    # Metrics
    st.header(f"Key Metrics for {year_filter}")
    total_sales = len(filtered_data)
    unique_items = filtered_data["Item Name"].nunique()
    total_revenue = filtered_data["Price"].sum()

    col1, col2, col3 = st.columns([1, 1, 1])
    col1.metric("Total Sales", total_sales)
    col2.metric("Unique Items Sold", unique_items)
    col3.metric("Total Revenue ($)", f"${total_revenue:,.2f}")

    # Sales Insights Section with 3 Columns
    st.header("Sales Insights")
    col1, col2, col3 = st.columns([1, 1, 1])  # Define three equal-width columns

    with col1:
        st.subheader("Sales by Category")
        dashboard_components_path_1 = None  # Initialize the path to None
        sales_by_category = filtered_data["Category"].value_counts()
        if sales_by_category.empty:
            st.warning("No data available for the selected filters.")
        else:
            fig, ax = plt.subplots(figsize=(6, 4))  # Uniform figure size
            sales_by_category.plot(kind="bar", ax=ax, color="skyblue")
            ax.set_title("Sales by Category")
            ax.set_ylabel("Number of Sales")
            ax.set_xlabel("Category")
            st.pyplot(fig)
            dashboard_components_path_1 = save_chart_to_file(fig,"sales_by_category.png")

    with col2:
        st.subheader("Sales by Item")
        dashboard_components_path_2 = None  # Initialize the path to None
        sales_by_item = filtered_data["Item Name"].value_counts()
        if sales_by_item.empty:
            st.warning("No data available for the selected filters.")
        else:
            fig, ax = plt.subplots(figsize=(6, 10))  # Uniform figure size
            sales_by_item.plot(kind="barh", ax=ax, color="lightgreen")
            ax.set_title("Sales by Item")
            ax.set_xlabel("Number of Sales")
            ax.set_ylabel("Item Name")
            st.pyplot(fig)
            dashboard_components_path_2 = save_chart_to_file(fig,"sales_by_item.png")

    with col3:
        st.subheader("Sales by Age Group")
        dashboard_components_path_3 = None  # Initialize the path to None

        # Define age groups
        age_bins = [0, 18, 30, 40, 50, np.inf]
        age_labels = ["Under 18", "18-30", "30-40", "40-50", "Above 50"]

        # Add a new column for age groups in the filtered data
        filtered_data["Age Group"] = pd.cut(
            filtered_data["Customer Age"], bins=age_bins, labels=age_labels, right=False
        )

        # Calculate sales by age group
        sales_by_age_group = filtered_data["Age Group"].value_counts().sort_index()
        sales_by_item = filtered_data["Item Name"].value_counts()
        sales_by_category = filtered_data["Category"].value_counts()

        if sales_by_item.empty or sales_by_category.empty or sales_by_age_group.empty:
            st.warning("No data available for the selected filters.")
        else:
            fig, ax = plt.subplots(figsize=(6, 5))  # Uniform figure size
            sales_by_age_group.plot(kind="bar", ax=ax, color="coral")
            ax.set_title("Sales by Age Group")
            ax.set_ylabel("Number of Sales")
            ax.set_xlabel("Age Group")
            st.pyplot(fig)
            dashboard_components_path_3 = save_chart_to_file(fig,"sales_by_age group.png")


    # Grid for Monthly Sales and Average Calories per Category
    st.header("Trend Analysis")
    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        st.subheader("Monthly Sales")
        dashboard_components_path_4 = None  # Initialize the path to None
        monthly_sales = filtered_data.groupby("Year-Month").size()
        if monthly_sales.empty:
            st.warning("No data available for the selected filters.")
        else:
            fig, ax = plt.subplots(figsize=(6, 6))  # Set uniform figure size
            monthly_sales.plot(kind="line", ax=ax, marker="o", color="purple")
            ax.set_title("Monthly Sales Trend")
            ax.set_ylabel("Number of Sales")
            ax.set_xlabel("Month")
            ax.grid(True, linestyle="--", alpha=0.6)
            st.pyplot(fig)
            dashboard_components_path_4 = save_chart_to_file(fig,"monthly_sales.png")

    with col2:
        st.subheader("Average Calories per Category")
        dashboard_components_path_5 = None  # Initialize the path to None
        calories_by_category = filtered_data.groupby("Category")["Calories"].mean()
        if calories_by_category.empty:
            st.warning("No data available for the selected filters.")
        else:
            fig, ax = plt.subplots(figsize=(6, 4))  # Set uniform figure size
            calories_by_category.plot(kind="bar", ax=ax, color="orange")
            ax.set_title("Average Calories per Category")
            ax.set_ylabel("Average Calories")
            ax.set_xlabel("Category")
            st.pyplot(fig)
            dashboard_components_path_5 = save_chart_to_file(fig,"avg_calories_per_category.png")

    with col3:
        monthly_sales = data.groupby(["Year", "Month", "Month Name", "Item Name"]).agg({"Price": "sum"}).reset_index()

        # Find the best-selling item for each month
        best_selling_items = monthly_sales.loc[monthly_sales.groupby(["Year", "Month"])["Price"].idxmax()]

        # Sort by Year and Month to ensure chronological order
        best_selling_items = best_selling_items.sort_values(["Year", "Month"]).drop(columns=["Month"])
        sales_by_item = filtered_data["Item Name"].value_counts()
        sales_by_category = filtered_data["Category"].value_counts()

        # Display the table in Streamlit
        st.subheader("Best-Selling Items Month-Wise")
        if sales_by_item.empty or sales_by_category.empty:
            st.warning("No data available for the selected filters.")
        else:
            st.table(best_selling_items.reset_index(drop=True))

    # Store paths for report generation
    st.session_state.dashboard_charts = [
        dashboard_components_path_1,
        dashboard_components_path_2,
        dashboard_components_path_3,
        dashboard_components_path_4,
        dashboard_components_path_5
    ]

    # Raw Data
    st.header(f"Raw Data for {year_filter}")
    sales_by_item = filtered_data["Item Name"].value_counts()
    sales_by_category = filtered_data["Category"].value_counts()
    if sales_by_item.empty or sales_by_category.empty:
        st.warning("Select appropriate item and category to see the data.")
    else:
        st.dataframe(filtered_data)

elif page == "Enhanced Predictions":
    st.title("Enhanced Predictions")

    # Sidebar Options for Category and Items
    st.sidebar.header("Prediction Options")
    selected_category = st.sidebar.selectbox("Select a Category:", options=data["Category"].unique())
    selected_item = st.sidebar.selectbox(
        "Select an Item:", options=data[data["Category"] == selected_category]["Item Name"].unique()
    )

    # Forecast Horizon
    forecast_horizon = st.sidebar.slider("Forecast Horizon (days):", min_value=7, max_value=90, value=30, step=7)

    # Aggregate data for the selected item
    item_sales = data[data["Item Name"] == selected_item]
    daily_sales = item_sales.groupby("Date").agg({"Price": "sum", "Temperature": "mean"}).reset_index()
    # Add Regressors: Simulate Academic Schedules and Weather Conditions
    daily_sales["academic_schedule"] = np.where(daily_sales["Date"].dt.month.isin([9, 10, 11, 12, 1, 2, 3, 4]), 1, 0)

    # Prepare data for Prophet
    df_prophet = daily_sales.rename(columns={"Date": "ds", "Price": "y", "Temperature": "temperature"})
    df_prophet["academic_schedule"] = daily_sales["academic_schedule"]

    # Initialize and fit the Prophet model with regressors
    st.write(f"Training the model...")
    model = Prophet()
    model.add_regressor("academic_schedule")
    model.add_regressor("temperature")
    model.fit(df_prophet)

    # Extend the forecast into the future
    future = model.make_future_dataframe(periods=forecast_horizon)
    future["academic_schedule"] = np.where(future["ds"].dt.month.isin([9, 10, 11, 12, 1, 2, 3, 4]), 1, 0)
    last_known_temp = daily_sales["Temperature"].iloc[-1]
    future["temperature"] = last_known_temp
    forecast = model.predict(future)

    # Display Trends and Seasonality
    st.header(f"Trends for '{selected_item}' item")
    fig = plot_components_plotly(model, forecast)
    st.plotly_chart(fig)

    # Category Trends
    st.header(f"Trends for '{selected_category}' Category")

    # Aggregate category data
    category_sales = data[data["Category"] == selected_category]
    daily_category_sales = category_sales.groupby("Date").agg({"Price": "sum", "Temperature": "mean"}).reset_index()
    daily_category_sales["academic_schedule"] = np.where(daily_category_sales["Date"].dt.month.isin([9, 10, 11, 12, 1, 2, 3, 4]), 1, 0)

    # Prepare data for Prophet
    df_category = daily_category_sales.rename(columns={"Date": "ds", "Price": "y", "Temperature": "temperature"})
    df_category["academic_schedule"] = daily_category_sales["academic_schedule"]
    category_model = Prophet()
    category_model.add_regressor("academic_schedule")
    category_model.add_regressor("temperature")
    category_model.fit(df_category)

    # Forecast category trends
    category_future = category_model.make_future_dataframe(periods=forecast_horizon)
    category_future["academic_schedule"] = np.where(category_future["ds"].dt.month.isin([9, 10, 11, 12, 1, 2, 3, 4]), 1, 0)
    last_known_temp = daily_category_sales["Temperature"].iloc[-1]
    category_future["temperature"] = last_known_temp
    category_forecast = category_model.predict(category_future)

    # Plot category trends
    st.write(f"Trend Analysis for '{selected_category}':")
    fig_category = plot_components_plotly(category_model, category_forecast)
    st.plotly_chart(fig_category)

    # Prophet Forecast Components Plot
    # st.header(f"Forecast Components for {selected_item}")
    fig1 = model.plot_components(forecast)
    # st.pyplot(fig1)

    # Save chart for PDF
    prophet_components_path = save_prophet_plot_to_file(fig1, "prophet_forecast_components.png")

    # Display Forecasted Data
    st.header("Forecasted Data based on academic schedule and temperature:")
    st.dataframe(forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]])

    # Predictions Forecast Plot
    st.header("Forecast based on academic schedule and temperature")
    fig2 = model.plot(forecast)
    st.plotly_chart(fig2)

    # Save chart for PDF
    prophet_forecast_path = save_prophet_plot_to_file(fig2, "prophet_forecast_plot.png")

    # Store paths for report generation
    st.session_state.prediction_charts = [prophet_components_path, prophet_forecast_path]

elif page == "Generate Report":
    st.title("Generate Report")
    st.write("This page allows you to generate a report containing the Dashboard and Predictions.")

    dashboard_charts = st.session_state.dashboard_charts
    prediction_charts = st.session_state.prediction_charts

    if st.button("Generate PDF Report"):
        report_path = generate_pdf_report(dashboard_charts, prediction_charts, output_file="sales_report.pdf")
        print("Generate Report")
        st.success(f"Report generated successfully: {report_path}")

        with open(report_path, "rb") as file:
            st.download_button(
                label="Download Report",
                data=file,
                file_name="sales_report.pdf",
                mime="application/pdf"
            )