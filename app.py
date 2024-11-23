import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from prophet import Prophet
from prophet.plot import plot_components_plotly
from datetime import datetime
import numpy as np
import time as time
import io

# Load the data
data = pd.read_csv("Updated_Sales_Data.csv")

# Convert the "Date" column to datetime format
data["Date"] = pd.to_datetime(data["Date"])
data["Year"] = data["Date"].dt.year
data["Year-Month"] = data["Date"].dt.to_period("M")

# Sidebar for Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Dashboard", "Enhanced Predictions"])

if page == "Dashboard":
    # Main Dashboard Logic
    st.title("Sales Dashboard")

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
        options=filtered_data_by_year["Item Name"].unique(),
        default=filtered_data_by_year["Item Name"].unique()
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

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Sales", total_sales)
    col2.metric("Unique Items Sold", unique_items)
    col3.metric("Total Revenue ($)", f"${total_revenue:,.2f}")

    # Sales by Category
    st.header("Sales by Category")
    sales_by_category = filtered_data["Category"].value_counts()

    if sales_by_category.empty:
        st.warning("No data available for the selected filters.")
    else:
        fig, ax = plt.subplots()
        sales_by_category.plot(kind="bar", ax=ax, color="skyblue")
        ax.set_title("Sales by Category")
        ax.set_ylabel("Number of Sales")
        ax.set_xlabel("Category")
        st.pyplot(fig)

    # Sales by Item
    st.header("Sales by Item")
    sales_by_item = filtered_data["Item Name"].value_counts()

    if sales_by_item.empty:
        st.warning("No data available for the selected filters.")
    else:
        fig, ax = plt.subplots(figsize=(10, 6))
        sales_by_item.plot(kind="barh", ax=ax, color="lightgreen")
        ax.set_title("Sales by Item")
        ax.set_xlabel("Number of Sales")
        ax.set_ylabel("Item Name")
        st.pyplot(fig)

    # Monthly Sales Line Chart
    st.header("Monthly Sales")
    monthly_sales = filtered_data.groupby("Year-Month").size()

    if monthly_sales.empty:
        st.warning("No data available for the selected filters.")
    else:
        fig, ax = plt.subplots()
        monthly_sales.plot(kind="line", ax=ax, marker="o", color="purple")
        ax.set_title("Monthly Sales Trend")
        ax.set_ylabel("Number of Sales")
        ax.set_xlabel("Month")
        ax.grid(True, linestyle="--", alpha=0.6)
        st.pyplot(fig)

    # Average Calories per Category
    st.header("Average Calories per Category")
    calories_by_category = filtered_data.groupby("Category")["Calories"].mean()

    if calories_by_category.empty:
        st.warning("No data available for the selected filters.")
    else:
        fig, ax = plt.subplots()
        calories_by_category.plot(kind="bar", ax=ax, color="orange")
        ax.set_title("Average Calories per Category")
        ax.set_ylabel("Average Calories")
        ax.set_xlabel("Category")
        st.pyplot(fig)

    # Raw Data
    st.header(f"Raw Data for {year_filter}")
    st.dataframe(filtered_data)

elif page == "Enhanced Predictions":
    # Enhanced Predictions Page
    st.title("Enhanced Predictions")

    # Sidebar Options for Category and Items
    st.sidebar.header("Prediction Options")
    selected_category = st.sidebar.selectbox(
        "Select a Category for Trend Analysis",
        options=data["Category"].unique()
    )

    selected_item = st.sidebar.selectbox(
        "Select an Item for Trend Analysis",
        options=data[data["Category"] == selected_category]["Item Name"].unique()
    )

    # Aggregate data for the selected item
    item_sales = data[data["Item Name"] == selected_item]
    daily_sales = item_sales.groupby("Date").agg({"Price": "sum"}).reset_index()

    # Add Regressors: Simulate Academic Schedules and Weather Conditions
    daily_sales["academic_schedule"] = np.where(daily_sales["Date"].dt.month.isin([9, 10, 11, 1, 2, 3]), 1, 0)  # Academic months
    daily_sales["temperature"] = 20 + 10 * np.sin(2 * np.pi * daily_sales["Date"].dt.month / 12)  # Simulated temperature

    # Prepare data for Prophet
    df_prophet = daily_sales.rename(columns={"Date": "ds", "Price": "y"})
    df_prophet["academic_schedule"] = daily_sales["academic_schedule"]
    df_prophet["temperature"] = daily_sales["temperature"]

    # Initialize and fit the Prophet model with regressors
    st.write(f"Training the model for '{selected_item}'...")
    model = Prophet()
    model.add_regressor("academic_schedule")
    model.add_regressor("temperature")
    model.fit(df_prophet)

    # Extend the forecast into the future
    future = model.make_future_dataframe(periods=30)  # Extend by 30 days
    future["academic_schedule"] = np.where(future["ds"].dt.month.isin([9, 10, 11, 1, 2, 3]), 1, 0)
    future["temperature"] = 20 + 10 * np.sin(2 * np.pi * future["ds"].dt.month / 12)

    forecast = model.predict(future)

    # Display Trends and Seasonality
    st.write(f"Forecast Components for '{selected_item}':")
    fig = plot_components_plotly(model, forecast)
    st.plotly_chart(fig)

    # Display Forecasted Data
    st.write("Forecasted Data:")
    st.dataframe(forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]])

    # Category Trends
    st.header(f"Trends for '{selected_category}' Category")

    # Aggregate category data
    category_sales = data[data["Category"] == selected_category]
    daily_category_sales = category_sales.groupby("Date").agg({"Price": "sum"}).reset_index()

    # Prepare data for Prophet
    df_category = daily_category_sales.rename(columns={"Date": "ds", "Price": "y"})
    category_model = Prophet()
    category_model.fit(df_category)

    # Forecast category trends
    category_future = category_model.make_future_dataframe(periods=30)
    category_forecast = category_model.predict(category_future)

    # Plot category trends
    st.write(f"Trend Analysis for '{selected_category}':")
    fig_category = plot_components_plotly(category_model, category_forecast)
    st.plotly_chart(fig_category)
